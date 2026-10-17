use crate::reasoning::{ReasoningArtifact, ReasoningEffort};
use crate::tool::{ToolCall, ToolResult};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::System => write!(f, "system"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Content {
    Text(String),
    Image(Image),
    ToolResult(ToolResult),
}

impl Content {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&Image> {
        match self {
            Content::Image(image) => Some(image),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResult> {
        match self {
            Content::ToolResult(result) => Some(result),
            _ => None,
        }
    }
}

/// An image in a message. Serialized as `{"Url": ..}` or `{"Base64":
/// {"data": .., "mime_type": ..}}`. Inside
/// [`with_image_store`](crate::image_store::with_image_store), a base64
/// image is saved in the store and written as a reference instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Image {
    Url(String),
    Base64 { data: String, mime_type: String },
}

impl Image {
    pub fn new_url(url: &str) -> Self {
        Image::Url(url.to_string())
    }

    pub fn new_base64(data: &str, mime_type: &str) -> Self {
        Image::Base64 {
            data: data.to_string(),
            mime_type: mime_type.to_string(),
        }
    }

    /// An image of the file content `bytes`, base64-encoded.
    pub fn from_bytes(bytes: &[u8], mime_type: &str) -> Self {
        use base64::Engine as _;
        Image::Base64 {
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
            mime_type: mime_type.to_string(),
        }
    }

    /// The image's width and height in pixels, read from the header of
    /// inline PNG, JPEG, GIF or WebP data. Decodes only the first 48 bytes and,
    /// for a JPEG, each segment's header up to its frame header, however much
    /// metadata comes before it. `None` for a
    /// URL or data whose size can't be read.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        let Image::Base64 { data, .. } = self else {
            return None;
        };
        let data = data.as_bytes();
        let head = decode_range(data, 0, 48)?;
        if head.starts_with(&[0xff, 0xd8]) {
            return jpeg_size(data);
        }
        image_size(&head)
    }

    /// An estimate of the input tokens the image takes in a request to a
    /// current Claude model (a token per 28 × 28 pixels, after scaling to
    /// 2,576 pixels on the long side and 4,784 tokens). Other models count
    /// images differently. [`Model::image_tokens`](crate::Model::image_tokens)
    /// gives a given model's count.
    pub fn estimated_tokens(&self) -> usize {
        crate::image_tokens::anthropic_tokens(
            self.dimensions().filter(|(w, h)| *w > 0 && *h > 0),
            crate::image_tokens::AnthropicTier::HIGH_RESOLUTION,
        )
    }
}

/// Bytes `start..end` of the base64 `data` (fewer at its end), decoding
/// only the characters that hold them: base64 decodes in groups of 4
/// characters from any multiple of 4. `None` when they aren't valid
/// standard base64.
fn decode_range(data: &[u8], start: usize, end: usize) -> Option<Vec<u8>> {
    use base64::Engine as _;
    let first = start / 3 * 4;
    let last = end.div_ceil(3).saturating_mul(4).min(data.len() / 4 * 4);
    if first >= last {
        return None;
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&data[first..last])
        .ok()?;
    let offset = start - first / 4 * 3;
    let end = (end - first / 4 * 3).min(decoded.len());
    decoded.get(offset..end).map(<[u8]>::to_vec)
}

/// The most JPEG segments [`jpeg_size`] walks before giving up.
const MAX_JPEG_SEGMENTS: usize = 1024;

/// A JPEG's width and height from its frame header, walking the segments
/// before it and decoding only each segment's header.
fn jpeg_size(data: &[u8]) -> Option<(u32, u32)> {
    let mut at = 2;
    for _ in 0..MAX_JPEG_SEGMENTS {
        let header = decode_range(data, at, at + 9)?;
        if header.len() < 4 || header[0] != 0xff {
            return None;
        }
        let marker = header[1];
        if marker == 0xff {
            // Fill byte.
            at += 1;
            continue;
        }
        let start_of_frame = matches!(marker, 0xc0..=0xcf) && !matches!(marker, 0xc4 | 0xc8 | 0xcc);
        if start_of_frame {
            let height = u16::from_be_bytes(header.get(5..7)?.try_into().ok()?);
            let width = u16::from_be_bytes(header.get(7..9)?.try_into().ok()?);
            return Some((u32::from(width), u32::from(height)));
        }
        if marker == 0xda {
            // The scan's data starts: no frame header came before it.
            return None;
        }
        at += 2 + usize::from(u16::from_be_bytes([header[2], header[3]]));
    }
    None
}

/// Width and height from an image file's first bytes: PNG, GIF or WebP.
fn image_size(bytes: &[u8]) -> Option<(u32, u32)> {
    let u16_le = |at: usize| Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?));
    let u24_le = |at: usize| {
        let b = bytes.get(at..at + 3)?;
        Some(u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16))
    };
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        let width = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
        let height = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
        return Some((width, height));
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some((u32::from(u16_le(6)?), u32::from(u16_le(8)?)));
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return match bytes.get(12..16)? {
            b"VP8 " => Some((
                u32::from(u16_le(26)? & 0x3fff),
                u32::from(u16_le(28)? & 0x3fff),
            )),
            b"VP8L" => {
                let bits = u32::from_le_bytes(bytes.get(21..25)?.try_into().ok()?);
                Some(((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1))
            }
            b"VP8X" => Some((u24_le(24)? + 1, u24_le(27)? + 1)),
            _ => None,
        };
    }
    None
}

/// A request setting changed from a point in the conversation onward,
/// without editing earlier turns, so the provider's prompt cache for the
/// turns before it stays valid. Added with
/// [`Conversation::add_reasoning_effort_update`].
///
/// [`Conversation::add_reasoning_effort_update`]: crate::conversation::Conversation::add_reasoning_effort_update
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationUpdate {
    /// Reasoning effort for the turns after this point.
    pub reasoning_effort: ReasoningEffort,
}

impl ConfigurationUpdate {
    pub fn reasoning_effort(effort: ReasoningEffort) -> Self {
        Self {
            reasoning_effort: effort,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
    /// Provider state an assistant message needs to be replayed, such as reasoning
    /// signatures, or for OpenAI the response's output items (see
    /// [`ReasoningArtifact::OpenAIOutputItems`]).
    pub reasoning_artifact: Option<ReasoningArtifact>,
    /// Tool calls produced by the model in this assistant message.
    /// Empty for user messages and assistant messages without tool use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub id: Option<String>,
    /// Set on a content-less system message that changes request settings
    /// from its position onward (see [`ConfigurationUpdate`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration_update: Option<ConfigurationUpdate>,
}

impl Message {
    pub fn new_user(text: &str) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_user_with_image(text: &str, image: Image) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::Text(text.to_string()), Content::Image(image)],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_user_with_contents(contents: Vec<Content>) -> Self {
        Self {
            role: Role::User,
            content: contents,
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_assistant(text: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_assistant_with_reasoning(text: &str, reasoning_artifact: ReasoningArtifact) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: Some(reasoning_artifact),
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    pub fn new_system(text: &str) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::Text(text.to_string())],
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: None,
        }
    }

    /// A content-less system message carrying a configuration update.
    pub fn new_configuration_update(update: ConfigurationUpdate) -> Self {
        Self {
            role: Role::System,
            content: Vec::new(),
            reasoning_artifact: None,
            tool_calls: Vec::new(),
            id: None,
            configuration_update: Some(update),
        }
    }

    /// The images this message holds, as content of their own and inside its
    /// tool results.
    pub fn image_count(&self) -> usize {
        self.content
            .iter()
            .map(|content| match content {
                Content::Image(_) => 1,
                Content::ToolResult(result) => result.images.len(),
                Content::Text(_) => 0,
            })
            .sum()
    }

    /// Whether this message holds an image, as content of its own or inside a
    /// tool result.
    pub fn has_images(&self) -> bool {
        self.image_count() > 0
    }

    /// Removes every image from this message, including image content and images
    /// inside its tool results (whose output text is kept, so each call and
    /// its result stay paired). Returns how many were removed. Text that
    /// labels an image, such as the runtime's `"Image from tool ..."`
    /// description before a sibling image, is left for the caller.
    pub fn remove_images(&mut self) -> usize {
        let before = self.image_count();
        self.content
            .retain(|content| !matches!(content, Content::Image(_)));
        for content in &mut self.content {
            if let Content::ToolResult(result) = content {
                result.images.clear();
            }
        }
        before
    }

    /// Returns `true` if this assistant message contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Whether this message is a configuration update rather than a turn.
    pub fn is_configuration_update(&self) -> bool {
        self.configuration_update.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolResult, ToolResultImage};

    #[test]
    fn images_inside_tool_results_count_and_are_removed_too() {
        let image = || Image::new_base64("cGl4ZWxz", "image/png");
        let mut message = Message::new_user_with_contents(vec![
            Content::ToolResult(
                ToolResult::new(Some("call_a".into()), "view_image", "a.png").with_images(vec![
                    ToolResultImage::new("a.png", image()),
                    ToolResultImage::new("", image()),
                ]),
            ),
            Content::ToolResult(ToolResult::new(Some("call_b".into()), "read", "text")),
            Content::Text("Image from tool view_image, call call_c: c.png".into()),
            Content::Image(image()),
        ]);
        assert!(message.has_images());
        assert_eq!(message.image_count(), 3);
        assert_eq!(message.remove_images(), 3);
        assert!(!message.has_images());
        // Results stay, with their output; label text is the caller's.
        assert_eq!(message.content.len(), 3);
        assert_eq!(message.content[0].as_tool_result().unwrap().output, "a.png");
        assert!(message.content[2].as_text().is_some());
        assert_eq!(message.remove_images(), 0);
        assert!(!Message::new_user("hi").has_images());
    }

    /// The size is read from each format's header, a JPEG's after the
    /// segments before its frame header, also far into the data.
    #[test]
    fn image_dimensions_come_from_the_header() {
        let image = |bytes: Vec<u8>| Image::from_bytes(&bytes, "image/x");
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        png.extend(640u32.to_be_bytes());
        png.extend(480u32.to_be_bytes());
        assert_eq!(image(png).dimensions(), Some((640, 480)));

        let mut gif = b"GIF89a".to_vec();
        gif.extend(320u16.to_le_bytes());
        gif.extend(200u16.to_le_bytes());
        assert_eq!(image(gif).dimensions(), Some((320, 200)));

        let webp = |chunk: &[u8], payload: &[u8]| {
            let mut bytes = b"RIFF\0\0\0\0WEBP".to_vec();
            bytes.extend(chunk);
            bytes.extend([0; 4]);
            bytes.extend(payload);
            bytes
        };
        let mut lossy = vec![0, 0, 0, 0x9d, 0x01, 0x2a];
        lossy.extend(100u16.to_le_bytes());
        lossy.extend(50u16.to_le_bytes());
        assert_eq!(image(webp(b"VP8 ", &lossy)).dimensions(), Some((100, 50)));
        let bits: u32 = 99 | (49 << 14);
        let mut lossless = vec![0x2f];
        lossless.extend(bits.to_le_bytes());
        assert_eq!(
            image(webp(b"VP8L", &lossless)).dimensions(),
            Some((100, 50))
        );
        let extended = [0, 0, 0, 0, 99, 0, 0, 49, 0, 0];
        assert_eq!(
            image(webp(b"VP8X", &extended)).dimensions(),
            Some((100, 50))
        );

        // A JPEG with 200 KB of metadata before its frame header.
        let mut jpeg = vec![0xff, 0xd8];
        for _ in 0..4 {
            jpeg.extend([0xff, 0xe1, 0xc3, 0x50]);
            jpeg.extend(vec![0; 0xc350 - 2]);
        }
        jpeg.extend([0xff, 0xc0, 0, 17, 8]);
        jpeg.extend(1080u16.to_be_bytes());
        jpeg.extend(1920u16.to_be_bytes());
        assert_eq!(image(jpeg).dimensions(), Some((1920, 1080)));

        // Segments of every length modulo 3 (each header at another place
        // in a base64 group), and fill bytes.
        let mut jpeg = vec![0xff, 0xd8];
        for length in [7u16, 8, 9, 10] {
            jpeg.extend([0xff, 0xe0]);
            jpeg.extend(length.to_be_bytes());
            jpeg.extend(vec![0xaa; usize::from(length) - 2]);
            jpeg.push(0xff);
        }
        jpeg.extend([0xff, 0xc2, 0, 17, 8]);
        jpeg.extend(600u16.to_be_bytes());
        jpeg.extend(800u16.to_be_bytes());
        jpeg.extend(vec![0x55; 100]);
        assert_eq!(image(jpeg.clone()).dimensions(), Some((800, 600)));
        // The scan starting before any frame header, or data cut short.
        let mut scan_first = vec![0xff, 0xd8, 0xff, 0xda, 0, 4, 0, 0];
        scan_first.extend(&jpeg[2..]);
        assert_eq!(image(scan_first).dimensions(), None);
        assert_eq!(image(jpeg[..40].to_vec()).dimensions(), None);
        // A format the reader doesn't know.
        let mut heic = b"\0\0\0\x18ftypheic".to_vec();
        heic.extend(vec![0; 1000]);
        assert_eq!(image(heic).dimensions(), None);

        assert_eq!(image(b"not an image".to_vec()).dimensions(), None);
        assert_eq!(Image::new_url("https://x/y.png").dimensions(), None);
        assert_eq!(Image::new_url("https://x/y.png").estimated_tokens(), 4_784);
    }

    /// Tokens follow the pixels, a token per 28 × 28 pixels after scaling
    /// down to a current Claude model's limit.
    #[test]
    fn image_tokens_follow_the_scaled_pixels() {
        let png = |width: u32, height: u32| {
            let mut bytes = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
            bytes.extend(width.to_be_bytes());
            bytes.extend(height.to_be_bytes());
            Image::from_bytes(&bytes, "image/png")
        };
        // A token per 28 × 28 pixels.
        assert_eq!(png(750, 100).estimated_tokens(), 27 * 4);
        assert_eq!(png(1, 1).estimated_tokens(), 1);
        assert_eq!(png(1500, 1000).estimated_tokens(), 54 * 36);
        // Scaled down to 4,784 tokens at most.
        let large = png(4000, 3000).estimated_tokens();
        assert!((4_700..=4_784).contains(&large), "{large}");
        // A long strip is scaled to 2,576 pixels long.
        assert_eq!(png(3136, 100).estimated_tokens(), 92 * 3);
    }
}
