//! Fitting images to what a provider takes. A request with an image over a
//! provider's limits fails, and a conversation sends its earlier images
//! with every request, so one oversized image would make every later
//! request of the conversation fail. [`Model::sendable_image`] shrinks such
//! an image (scaled down to the longest side the provider takes, keeping
//! its aspect, and re-encoded until it is small enough) and refuses, with
//! the reason, only one it can't read or shrink enough.

use base64::Engine as _;

use crate::message::Image;
use crate::model::{Model, Provider};

/// What a model's provider takes of images, as its documentation gives it,
/// with some room: inline images, by their base64 data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLimits {
    /// The most bytes of one image's base64 data.
    pub max_image_bytes: usize,
    /// The most pixels on an image's longer side.
    pub max_long_edge: u32,
    /// The most images one request holds.
    pub max_images: usize,
    /// The most bytes of base64 image data one request holds, leaving room
    /// for its text within the provider's request size limit.
    pub max_request_image_bytes: usize,
}

const MIB: usize = 1024 * 1024;

/// Anthropic's: 5 MB per image, at most 2,000 pixels a side once a request
/// holds more than 20 images (8,000 otherwise: fitting every image to 2,000
/// keeps the requests valid as the conversation grows), 100 images, and
/// 32 MB per request.
const ANTHROPIC: ImageLimits = ImageLimits {
    max_image_bytes: 5 * MIB,
    max_long_edge: 2_000,
    max_images: 100,
    max_request_image_bytes: 24 * MIB,
};

/// OpenAI's: 50 MB per request and 500 images. Larger images are scaled
/// down by the API to fit 2,048 pixels.
const OPENAI: ImageLimits = ImageLimits {
    max_image_bytes: 20 * MIB,
    max_long_edge: 8_000,
    max_images: 500,
    max_request_image_bytes: 40 * MIB,
};

/// Google's: 20 MB of inline data per request.
const GOOGLE: ImageLimits = ImageLimits {
    max_image_bytes: 15 * MIB,
    max_long_edge: 8_000,
    max_images: 3_000,
    max_request_image_bytes: 15 * MIB,
};

impl Model {
    /// What this model's provider takes of images ([`ImageLimits`]):
    /// Anthropic's for Anthropic and Claude Code models and OpenRouter's
    /// `anthropic/` ones, OpenAI's and Google's likewise, and Anthropic's,
    /// the strictest, for other OpenRouter models.
    pub fn image_limits(&self) -> ImageLimits {
        match self.provider {
            Provider::Anthropic | Provider::ClaudeCode => ANTHROPIC,
            Provider::OpenAI => OPENAI,
            Provider::Google => GOOGLE,
            Provider::OpenRouter => match self.id.split_once('/').map(|(vendor, _)| vendor) {
                Some("openai") => OPENAI,
                Some("google") => GOOGLE,
                _ => ANTHROPIC,
            },
        }
    }

    /// `image` as this model can be sent it: `None` when it can be sent as
    /// it is, the image shrunk to the model's [`image_limits`](Self::image_limits)
    /// otherwise (see [`fit_image`]), or why it can't be sent (the reasons
    /// of [`image_refusal`](Self::image_refusal), or an image too large that
    /// can't be read or shrunk enough).
    pub fn sendable_image(&self, image: &Image) -> Result<Option<Image>, String> {
        if let Some(reason) = self.image_refusal(image) {
            return Err(reason);
        }
        fit_image(image, &self.image_limits())
    }
}

/// `image` fitted to `limits`: `None` when it fits as it is (and for an
/// image by URL, whose size isn't known here), else the image scaled down
/// to [`max_long_edge`](ImageLimits::max_long_edge) (keeping its aspect),
/// and further while its data is over
/// [`max_image_bytes`](ImageLimits::max_image_bytes). A PNG stays a PNG
/// when that is small enough (it keeps text sharp), otherwise it becomes a
/// JPEG (transparency on white), as does every other format. PNG, JPEG,
/// GIF (its first frame) and WebP can be read. An image that can't be read
/// or shrunk below the limit is an error saying why.
pub fn fit_image(image: &Image, limits: &ImageLimits) -> Result<Option<Image>, String> {
    let Image::Base64 { data, mime_type } = image else {
        return Ok(None);
    };
    let size = image.dimensions();
    let too_wide = size.is_some_and(|(width, height)| width.max(height) > limits.max_long_edge);
    if data.len() <= limits.max_image_bytes && !too_wide {
        return Ok(None);
    }
    let over = || {
        if data.len() > limits.max_image_bytes {
            format!(
                "its {:.1} MB are over the {:.1} MB a request takes per image",
                data.len() as f64 / MIB as f64,
                limits.max_image_bytes as f64 / MIB as f64
            )
        } else {
            let (width, height) = size.unwrap_or_default();
            format!(
                "its {width}×{height} pixels are over the {} a side the model takes",
                limits.max_long_edge
            )
        }
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|_| format!("{}, and its data isn't valid base64", over()))?;
    let decoded = image::load_from_memory(&bytes).map_err(|error| {
        format!(
            "{}, and its {mime_type} data can't be read to shrink it ({error}): convert it to PNG or JPEG",
            over()
        )
    })?;
    let png = mime_type.trim().eq_ignore_ascii_case("image/png");
    let (width, height) = (decoded.width(), decoded.height());
    let mut scale = (f64::from(limits.max_long_edge) / f64::from(width.max(height))).min(1.0);
    for _ in 0..8 {
        let resized = if scale < 1.0 {
            let fit = |side: u32| ((f64::from(side) * scale).round() as u32).max(1);
            decoded.resize_exact(
                fit(width),
                fit(height),
                image::imageops::FilterType::Triangle,
            )
        } else {
            decoded.clone()
        };
        if png
            && let Some(encoded) = encode_png(&resized)
            && base64_len(encoded.len()) <= limits.max_image_bytes
        {
            return Ok(Some(Image::from_bytes(&encoded, "image/png")));
        }
        if let Some(encoded) = encode_jpeg(&resized)
            && base64_len(encoded.len()) <= limits.max_image_bytes
        {
            return Ok(Some(Image::from_bytes(&encoded, "image/jpeg")));
        }
        scale *= 0.7;
    }
    Err(format!("{}, and it can't be shrunk enough", over()))
}

/// The length of the base64 of `bytes` bytes.
fn base64_len(bytes: usize) -> usize {
    bytes.div_ceil(3) * 4
}

fn encode_png(image: &image::DynamicImage) -> Option<Vec<u8>> {
    let mut encoded = Vec::new();
    image
        .write_to(
            &mut std::io::Cursor::new(&mut encoded),
            image::ImageFormat::Png,
        )
        .ok()?;
    Some(encoded)
}

/// JPEG at quality 85, transparency laid on white.
fn encode_jpeg(image: &image::DynamicImage) -> Option<Vec<u8>> {
    let rgba = image.to_rgba8();
    let mut rgb = image::RgbImage::new(rgba.width(), rgba.height());
    for (target, source) in rgb.pixels_mut().zip(rgba.pixels()) {
        let alpha = u32::from(source[3]);
        for channel in 0..3 {
            let value = (u32::from(source[channel]) * alpha + 255 * (255 - alpha)) / 255;
            target[channel] = value as u8;
        }
    }
    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 85)
        .encode_image(&rgb)
        .ok()?;
    Some(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32, noise: bool) -> Image {
        let mut seed = 1u32;
        let pixels = image::RgbaImage::from_fn(width, height, |x, y| {
            if noise {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let [a, b, c, _] = seed.to_le_bytes();
                image::Rgba([a, b, c, 255])
            } else {
                image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255])
            }
        });
        let encoded = encode_png(&image::DynamicImage::ImageRgba8(pixels)).unwrap();
        Image::from_bytes(&encoded, "image/png")
    }

    fn limits(max_image_bytes: usize, max_long_edge: u32) -> ImageLimits {
        ImageLimits {
            max_image_bytes,
            max_long_edge,
            max_images: 100,
            max_request_image_bytes: 100 * MIB,
        }
    }

    /// An image within the limits is sent as it is.
    #[test]
    fn an_image_within_the_limits_is_left_alone() {
        let image = png(40, 30, false);
        assert_eq!(fit_image(&image, &limits(MIB, 100)), Ok(None));
        assert_eq!(
            fit_image(&Image::new_url("https://x/y.png"), &limits(1, 1)),
            Ok(None)
        );
    }

    /// Too wide: scaled down to the longest side, keeping its aspect, and
    /// still a PNG.
    #[test]
    fn a_wide_image_is_scaled_to_the_longest_side() {
        let image = png(400, 100, false);
        let fitted = fit_image(&image, &limits(MIB, 200)).unwrap().unwrap();
        assert_eq!(fitted.dimensions(), Some((200, 50)));
        assert!(matches!(&fitted, Image::Base64 { mime_type, .. } if mime_type == "image/png"));
    }

    /// Too large in bytes: re-encoded as JPEG, and scaled down further until
    /// it fits.
    #[test]
    fn a_large_image_is_reencoded_and_shrunk_until_it_fits() {
        let image = png(300, 300, true);
        let Image::Base64 { data, .. } = &image else {
            unreachable!()
        };
        let limit = data.len() / 10;
        let fitted = fit_image(&image, &limits(limit, 1_000)).unwrap().unwrap();
        let Image::Base64 { data, mime_type } = &fitted else {
            unreachable!()
        };
        assert!(data.len() <= limit, "{} > {limit}", data.len());
        assert_eq!(mime_type, "image/jpeg");
        let (width, height) = fitted.dimensions().unwrap();
        assert_eq!(width, height);
    }

    /// What can't be read or shrunk enough is refused, saying why.
    #[test]
    fn what_cant_be_read_or_shrunk_is_refused() {
        let heic = Image::from_bytes(&[7; 3000], "image/heic");
        let error = fit_image(&heic, &limits(1000, 1_000)).unwrap_err();
        assert!(error.contains("a request takes per image"), "{error}");
        assert!(error.contains("image/heic data can't be read"), "{error}");
        let error = fit_image(&png(300, 300, true), &limits(8, 1_000)).unwrap_err();
        assert!(error.contains("can't be shrunk enough"), "{error}");
    }

    /// Each provider's limits; OpenRouter's follow the model's maker.
    #[test]
    fn limits_follow_the_provider() {
        assert_eq!(Model::anthropic_claude_opus_5_5().image_limits(), ANTHROPIC);
        assert_eq!(Model::openai_gpt_5_6_luna().image_limits(), OPENAI);
        assert_eq!(
            Model::openrouter("google/gemini-3.5-flash").image_limits(),
            GOOGLE
        );
        assert_eq!(Model::openrouter("meta/llama-5").image_limits(), ANTHROPIC);
        let wide = png(2_500, 10, false);
        let fitted = Model::anthropic_claude_opus_5_5()
            .sendable_image(&wide)
            .unwrap()
            .unwrap();
        assert_eq!(fitted.dimensions(), Some((2_000, 8)));
        assert_eq!(Model::openai_gpt_5_6_luna().sendable_image(&wide), Ok(None));
    }
}
