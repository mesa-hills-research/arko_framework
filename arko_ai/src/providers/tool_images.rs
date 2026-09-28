//! Tool-result images for models that don't take them inside a result
//! ([`Model::accepts_tool_result_images`] is false): they follow all of the
//! turn's results, in the layout the arko_agents runtime has always used for
//! tool images, so requests for these models are unchanged.
//!
//! [`Model::accepts_tool_result_images`]: crate::model::Model::accepts_tool_result_images

use crate::{
    message::{Content, Message},
    tool::{ToolResult, ToolResultImage},
};
use std::borrow::Cow;

/// The text sent before a result image that follows the turn's results: the
/// text the arko_agents runtime writes before tool images.
pub(crate) fn fallback_label(result: &ToolResult, image: &ToolResultImage) -> String {
    format!(
        "Image from tool {}, call {}: {}",
        result.name,
        result.call_id.as_deref().unwrap_or("unidentified"),
        image.label
    )
}

/// The images of `message`'s tool results, each after its
/// [`fallback_label`], in order.
pub(crate) fn moved_result_images(message: &Message) -> Vec<Content> {
    message
        .content
        .iter()
        .filter_map(Content::as_tool_result)
        .flat_map(|result| {
            result.images.iter().flat_map(move |image| {
                [
                    Content::Text(fallback_label(result, image)),
                    Content::Image(image.image.clone()),
                ]
            })
        })
        .collect()
}

/// A message's content other than its tool results, in order, followed, when
/// the images can't go `inline` in the results, by the results' images (see
/// [`moved_result_images`]).
pub(crate) fn content_after_results(message: &Message, inline: bool) -> Vec<Cow<'_, Content>> {
    let mut rest: Vec<Cow<'_, Content>> = message
        .content
        .iter()
        .filter(|content| content.as_tool_result().is_none())
        .map(Cow::Borrowed)
        .collect();
    if !inline {
        rest.extend(moved_result_images(message).into_iter().map(Cow::Owned));
    }
    rest
}
