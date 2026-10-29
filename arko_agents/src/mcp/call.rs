//! A tool call on an MCP server, and its result as a tool's output: shared
//! by [`McpTool`](super::McpTool) and [`McpHub`](super::McpHub).

use serde_json::Value;

use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CancelledNotification, CancelledNotificationParam,
    ClientRequest, ErrorCode, RawContent, RequestId, ResourceContents, ServerResult,
};
use rmcp::service::{Peer, PeerRequestOptions, RoleClient, ServiceError};

use crate::tool::{ToolError, ToolImage, ToolOutput};

/// The image types accepted from MCP tools.
const IMAGE_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];

/// Why a call failed: the tool error, and the server's message when it
/// rejected the arguments (JSON-RPC's "invalid params", `-32602`).
#[derive(Debug)]
pub(crate) struct CallFailure {
    pub(crate) error: ToolError,
    pub(crate) invalid_params: Option<String>,
}

impl From<ToolError> for CallFailure {
    fn from(error: ToolError) -> Self {
        Self {
            error,
            invalid_params: None,
        }
    }
}

/// Calls `tool` on `server` (named for errors) through `peer`, and returns
/// the call's text and its images (MCP `image` content, and embedded image
/// resources) as the result's images. Other content is described in the
/// text. A resource's text is included, and audio, other binary resources
/// and images of a type the providers don't take are named. A result the
/// server marks `isError` is a tool error with the server's text, and
/// structured content is printed when the result has no text. Dropping the
/// call before its answer cancels the request on the server.
pub(crate) async fn call_tool(
    peer: &Peer<RoleClient>,
    server: &str,
    tool: &str,
    arguments: serde_json::Map<String, Value>,
) -> Result<ToolOutput, CallFailure> {
    let call_params = CallToolRequestParams::new(tool.to_string()).with_arguments(arguments);

    let failed = |e: ServiceError| {
        let invalid_params = match &e {
            ServiceError::McpError(error) if error.code == ErrorCode::INVALID_PARAMS => {
                Some(error.message.to_string())
            }
            _ => None,
        };
        CallFailure {
            error: ToolError::ExecutionFailed(format!(
                "MCP tool '{tool}' on server '{server}' call failed: {e}"
            )),
            invalid_params,
        }
    };
    let request = ClientRequest::CallToolRequest(CallToolRequest::new(call_params));
    let handle = peer
        .send_cancellable_request(request, PeerRequestOptions::no_options())
        .await
        .map_err(failed)?;
    let mut cancel = CancelOnDrop {
        peer: peer.clone(),
        request: Some(handle.id.clone()),
    };
    let response = handle.await_response().await;
    // Answered, or the server is gone: there's nothing to cancel.
    cancel.request = None;
    let result = match response.map_err(failed)? {
        ServerResult::CallToolResult(result) => result,
        _ => return Err(failed(ServiceError::UnexpectedResponse)),
    };

    let mut output = ToolOutput::default();
    let mut push = |text: &str| {
        if !output.text.is_empty() {
            output.text.push('\n');
        }
        output.text.push_str(text);
    };
    let mut images = Vec::new();
    for content in &result.content {
        match &content.raw {
            RawContent::Text(text) => push(&text.text),
            RawContent::Image(image) => match image_type(&image.mime_type) {
                Some(mime_type) => images.push(ToolImage::base64(
                    format!("image {} ({mime_type})", images.len() + 1),
                    image.data.clone(),
                    mime_type,
                )),
                None => push(&format!(
                    "[An image of type {} that the model can't be shown.]",
                    image.mime_type
                )),
            },
            RawContent::Resource(resource) => match &resource.resource {
                ResourceContents::TextResourceContents { uri, text, .. } => {
                    push(&format!("Resource {uri}:\n{text}"))
                }
                ResourceContents::BlobResourceContents {
                    uri,
                    mime_type,
                    blob,
                    ..
                } => match mime_type.as_deref().and_then(image_type) {
                    Some(mime_type) => {
                        images.push(ToolImage::base64(uri.clone(), blob.clone(), mime_type))
                    }
                    None => push(&format!(
                        "[Resource {uri} has binary content ({}) that can't be shown.]",
                        mime_type.as_deref().unwrap_or("of an unknown type")
                    )),
                },
            },
            RawContent::Audio(audio) => push(&format!(
                "[Audio ({}) that the model can't be given.]",
                audio.mime_type
            )),
            RawContent::ResourceLink(link) => {
                push(&format!("Resource link: {} ({})", link.uri, link.name))
            }
        }
    }
    output.images = images;

    if result.is_error == Some(true) {
        return Err(ToolError::ExecutionFailed(if output.text.is_empty() {
            format!("MCP tool '{tool}' returned an error with no text content")
        } else {
            output.text
        })
        .into());
    }

    // Descriptions synthesized for non-text content must not hide the
    // server's structured result when it supplied no text equivalent.
    let has_text = result
        .content
        .iter()
        .any(|content| matches!(&content.raw, RawContent::Text(text) if !text.text.is_empty()));
    if !has_text && let Some(ref structured) = result.structured_content {
        if !output.text.is_empty() {
            output.text.push_str("\n\n");
        }
        output
            .text
            .push_str(&serde_json::to_string_pretty(structured).unwrap_or_default());
    }

    Ok(output)
}

/// Cancels a request on the server (MCP's `notifications/cancelled`) when
/// the call awaiting it is dropped first: it was stopped, ran past its time
/// limit, or its task was cancelled. The server can then stop working on
/// it, and a server that handles one request at a time takes the next.
struct CancelOnDrop {
    peer: Peer<RoleClient>,
    request: Option<RequestId>,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let Some(request_id) = self.request.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let peer = self.peer.clone();
        runtime.spawn(async move {
            let notification = CancelledNotification::new(CancelledNotificationParam {
                request_id,
                reason: Some("The call was stopped.".into()),
            });
            let _ = peer.send_notification(notification.into()).await;
        });
    }
}

/// `mime_type` as one of [`IMAGE_TYPES`] (`image/jpg` read as JPEG), or
/// `None` for a type the providers don't take.
fn image_type(mime_type: &str) -> Option<&'static str> {
    let mime_type = mime_type.trim().to_ascii_lowercase();
    let mime_type = if mime_type == "image/jpg" {
        "image/jpeg".to_string()
    } else {
        mime_type
    };
    IMAGE_TYPES.into_iter().find(|known| *known == mime_type)
}
