//! Fitting images to what a provider takes. A request with an image over a
//! provider's limits, or of a format it doesn't take, fails, and a
//! conversation sends its earlier images with every request, so one such
//! image would make every later request of the conversation fail.
//! [`Model::sendable_image`] converts or shrinks such an image (turned
//! upright, scaled down to what the provider takes, keeping its aspect, and
//! re-encoded until it is small enough) and refuses, with the reason, only
//! one it can't read or shrink enough.
//!
//! The decoding and encoding run on a blocking thread when called from a
//! Tokio worker ([`tokio::task::block_in_place`]), and a decode is limited
//! to [`MAX_DECODE_BYTES`], so a small file of enormous size is refused
//! quickly instead of taking seconds and gigabytes.

use std::io::Cursor;

use base64::Engine as _;
use image::metadata::Orientation;
use image::{AnimationDecoder as _, DynamicImage, ImageDecoder as _, ImageFormat, ImageReader};

use crate::message::Image;
use crate::model::{Model, Provider};

/// Limits on inline images' base64 data, from the model provider's
/// documentation with some room.
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
    /// The most patches an image may cover, for a provider that refuses
    /// an image over it instead of shrinking it.
    pub max_patches: Option<PatchLimit>,
    /// The image formats the provider takes, as MIME types.
    pub formats: &'static [&'static str],
    /// Whether the provider takes an animated image (using its first frame).
    pub animations: bool,
}

/// A limit on the squares of `size` pixels an image covers, with
/// `⌈width / size⌉ × ⌈height / size⌉` at most `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchLimit {
    pub size: u32,
    pub max: usize,
}

impl PatchLimit {
    /// The patches a `width` × `height` image covers.
    pub fn patches(&self, width: u32, height: u32) -> usize {
        width.div_ceil(self.size) as usize * height.div_ceil(self.size) as usize
    }
}

const MIB: usize = 1024 * 1024;

/// The most memory decoding one image may take. An image larger than this
/// once decoded is too large to shrink.
pub const MAX_DECODE_BYTES: u64 = 256 * 1024 * 1024;

/// Anthropic's: 10 MB of base64 per image on its own API and 5 MB on other
/// clouds, 8,000 pixels a side and 2,000 once a request holds more than 20
/// images (fitting every image to 2,000 keeps the requests valid as the
/// conversation grows), 100 images for models with a 200k-token context
/// and 600 for the others, and 32 MB per request. JPEG, PNG, GIF and WebP,
/// an animation's first frame.
const ANTHROPIC: ImageLimits = ImageLimits {
    max_image_bytes: 5 * MIB,
    max_long_edge: 2_000,
    max_images: 100,
    max_request_image_bytes: 24 * MIB,
    max_patches: None,
    formats: &["image/jpeg", "image/png", "image/gif", "image/webp"],
    animations: true,
};

/// OpenAI's: an image over 30,000 patches of 32 pixels is refused, not
/// shrunk (the default detail of current models keeps an image's size), 512
/// MB per request and 1,500 images. PNG, JPEG, WebP and GIF without
/// animation.
const OPENAI: ImageLimits = ImageLimits {
    max_image_bytes: 20 * MIB,
    max_long_edge: 65_535,
    max_images: 500,
    max_request_image_bytes: 40 * MIB,
    max_patches: Some(PatchLimit {
        size: 32,
        max: 30_000,
    }),
    formats: &["image/png", "image/jpeg", "image/webp", "image/gif"],
    animations: false,
};

/// Google's: 20 MB of inline data per request and 3,600 images. PNG, JPEG,
/// WebP, HEIC and HEIF.
const GOOGLE: ImageLimits = ImageLimits {
    max_image_bytes: 15 * MIB,
    max_long_edge: 8_000,
    max_images: 3_000,
    max_request_image_bytes: 15 * MIB,
    max_patches: None,
    formats: &[
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    ],
    animations: false,
};

impl Model {
    /// The image limits for this model's provider ([`ImageLimits`]). Uses
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

    /// `image` as this model can be sent it. Returns `None` when it can be sent as
    /// it is, the image converted or shrunk to the model's
    /// [`image_limits`](Self::image_limits) otherwise (see [`fit_image`]),
    /// or why it can't be sent (the reasons of
    /// [`image_refusal`](Self::image_refusal), or an image that can't be
    /// read, converted or shrunk enough).
    pub fn sendable_image(&self, image: &Image) -> Result<Option<Image>, String> {
        if let Some(reason) = self.image_refusal(image) {
            return Err(reason);
        }
        fit_image(image, &self.image_limits())
    }

    /// Whether [`sendable_image`](Self::sendable_image) would send `image`,
    /// as it is or fitted, judged from its header without decoding it. This
    /// is cheap enough to check every recipient of an image before fitting it
    /// for one. `Err` with the reason it can't go.
    pub fn check_image(&self, image: &Image) -> Result<(), String> {
        if let Some(reason) = self.image_refusal(image) {
            return Err(reason);
        }
        let limits = self.image_limits();
        let Image::Base64 { data, mime_type } = image else {
            return Ok(());
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data.as_bytes()) else {
            return unreadable_as_is(data, mime_type, &limits).map(|_| ());
        };
        let header = Header::read(&bytes, mime_type, image.dimensions());
        match header.work(data, &limits) {
            Work::None => Ok(()),
            Work::Fit(why) => match (header.format, header.size) {
                (None, _) => Err(why.unreadable(mime_type)),
                (Some(_), Some((width, height)))
                    if u64::from(width) * u64::from(height) * 4 > MAX_DECODE_BYTES =>
                {
                    Err(too_large(width, height))
                }
                _ => Ok(()),
            },
        }
    }
}

/// `image` fitted to `limits`. Returns `None` when it can go as it is (and for an
/// image by URL, whose size isn't known here). Otherwise the image decoded,
/// turned upright (its EXIF orientation applied), scaled down to
/// [`max_long_edge`](ImageLimits::max_long_edge) and
/// [`max_patches`](ImageLimits::max_patches) (keeping its aspect), and
/// further while its data is over
/// [`max_image_bytes`](ImageLimits::max_image_bytes). A lossless image
/// (PNG, GIF, BMP, TIFF) becomes a PNG when that is small enough, otherwise
/// a JPEG (transparency on white), as does every other format. An image of
/// a format the provider doesn't take is converted, an animation to its
/// first frame. PNG, JPEG, GIF, WebP, BMP and TIFF can be read. An image
/// that can't be read, or shrunk below the limit, is an error saying why.
pub fn fit_image(image: &Image, limits: &ImageLimits) -> Result<Option<Image>, String> {
    let Image::Base64 { data, mime_type } = image else {
        return Ok(None);
    };
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data.as_bytes()) else {
        return unreadable_as_is(data, mime_type, limits);
    };
    let header = Header::read(&bytes, mime_type, image.dimensions());
    let why = match header.work(data, limits) {
        Work::None => return Ok(None),
        Work::Fit(why) => why,
    };
    let Some(format) = header.format else {
        return Err(why.unreadable(mime_type));
    };
    if let Some((width, height)) = header.size
        && u64::from(width) * u64::from(height) * 4 > MAX_DECODE_BYTES
    {
        return Err(format!(
            "{}, and {}",
            why.describe(),
            too_large(width, height)
        ));
    }
    off_the_runtime(|| fit_decoded(&bytes, format, limits, &why))
}

/// Runs `work` without holding up a Tokio worker: through `block_in_place`
/// on a multi-threaded runtime's worker, directly otherwise.
fn off_the_runtime<T>(work: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(work)
        }
        _ => work(),
    }
}

/// Data that isn't standard base64 can't be checked: it goes as it is when
/// it is within the byte limit and of a format the provider takes.
fn unreadable_as_is(
    data: &str,
    mime_type: &str,
    limits: &ImageLimits,
) -> Result<Option<Image>, String> {
    let mime = normalized(mime_type);
    if data.len() > limits.max_image_bytes {
        return Err(format!(
            "{}, and its data isn't valid base64",
            Why::Bytes(data.len(), limits.max_image_bytes).describe()
        ));
    }
    if !limits.formats.contains(&mime.as_str()) {
        return Err(Why::Format(mime).unreadable(mime_type));
    }
    Ok(None)
}

/// What the header of an image's bytes says.
struct Header {
    /// The format its bytes are in, if it is one that can be read here.
    format: Option<ImageFormat>,
    size: Option<(u32, u32)>,
    orientation: Orientation,
    animated: bool,
    /// The MIME type it is sent as: its format's, or the one it came with.
    mime: String,
}

impl Header {
    /// The header of `bytes`, sent as `mime_type`, whose size the image's
    /// own header reader found as `size` (for a decoder that can't start).
    fn read(bytes: &[u8], mime_type: &str, size: Option<(u32, u32)>) -> Self {
        let format = image::guess_format(bytes)
            .ok()
            .filter(|format| readable(*format));
        let decoder = format.and_then(|format| {
            ImageReader::with_format(Cursor::new(bytes), format)
                .into_decoder()
                .ok()
        });
        let (size, orientation) = match decoder {
            Some(mut decoder) => (
                Some(decoder.dimensions()),
                decoder.orientation().unwrap_or(Orientation::NoTransforms),
            ),
            None => (size, Orientation::NoTransforms),
        };
        let animated = match format {
            Some(ImageFormat::Gif) => image::codecs::gif::GifDecoder::new(Cursor::new(bytes))
                .is_ok_and(|decoder| decoder.into_frames().take(2).count() > 1),
            Some(ImageFormat::WebP) => image::codecs::webp::WebPDecoder::new(Cursor::new(bytes))
                .is_ok_and(|decoder| decoder.has_animation()),
            _ => false,
        };
        Self {
            format,
            size,
            orientation,
            animated,
            mime: format.map_or_else(
                || normalized(mime_type),
                |format| format.to_mime_type().to_string(),
            ),
        }
    }

    /// Whether the image goes as it is, or why it must be fitted.
    fn work(&self, data: &str, limits: &ImageLimits) -> Work {
        if !limits.formats.contains(&self.mime.as_str()) {
            return Work::Fit(Why::Format(self.mime.clone()));
        }
        if self.animated && !limits.animations {
            return Work::Fit(Why::Animated(self.mime.clone()));
        }
        if data.len() > limits.max_image_bytes {
            return Work::Fit(Why::Bytes(data.len(), limits.max_image_bytes));
        }
        if let Some((width, height)) = self.size {
            if width.max(height) > limits.max_long_edge {
                return Work::Fit(Why::Edge(width, height, limits.max_long_edge));
            }
            if let Some(patches) = limits.max_patches
                && patches.patches(width, height) > patches.max
            {
                return Work::Fit(Why::Patches(width, height, patches));
            }
        }
        if self.orientation != Orientation::NoTransforms {
            return Work::Fit(Why::Orientation);
        }
        Work::None
    }
}

enum Work {
    None,
    Fit(Why),
}

/// Why an image can't go as it is.
enum Why {
    Format(String),
    Animated(String),
    Bytes(usize, usize),
    Edge(u32, u32, u32),
    Patches(u32, u32, PatchLimit),
    Orientation,
}

impl Why {
    fn describe(&self) -> String {
        let mb = |bytes: usize| bytes as f64 / MIB as f64;
        match self {
            Why::Format(mime) => format!("the model doesn't take {mime} images"),
            Why::Animated(mime) => format!("the model doesn't take animated {mime} images"),
            Why::Bytes(bytes, max) => format!(
                "its {:.1} MB are over the {:.1} MB a request takes per image",
                mb(*bytes),
                mb(*max)
            ),
            Why::Edge(width, height, max) => {
                format!("its {width}×{height} pixels are over the {max} a side the model takes")
            }
            Why::Patches(width, height, limit) => format!(
                "its {width}×{height} pixels cover {} squares of {} pixels, over the {} the model takes",
                limit.patches(*width, *height),
                limit.size,
                limit.max
            ),
            Why::Orientation => "it must be turned upright".to_string(),
        }
    }

    /// The reason for an image that needs fitting but can't be read here.
    fn unreadable(&self, mime_type: &str) -> String {
        format!(
            "{}, and its {} data can't be read here to convert it. Convert it to PNG or JPEG",
            self.describe(),
            normalized(mime_type)
        )
    }
}

fn too_large(width: u32, height: u32) -> String {
    format!(
        "its {width}×{height} pixels are too large to shrink (over {} MB once decoded)",
        MAX_DECODE_BYTES / (1024 * 1024)
    )
}

/// The formats read here.
fn readable(format: ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Png
            | ImageFormat::Jpeg
            | ImageFormat::Gif
            | ImageFormat::WebP
            | ImageFormat::Bmp
            | ImageFormat::Tiff
    )
}

/// `mime_type` trimmed, lowercase, `image/jpg` as `image/jpeg`.
fn normalized(mime_type: &str) -> String {
    match mime_type.trim().to_ascii_lowercase().as_str() {
        "image/jpg" => "image/jpeg".to_string(),
        mime => mime.to_string(),
    }
}

/// Decodes `bytes` (of `format`), turns it upright, and shrinks and
/// re-encodes it until it fits `limits`.
fn fit_decoded(
    bytes: &[u8],
    format: ImageFormat,
    limits: &ImageLimits,
    why: &Why,
) -> Result<Option<Image>, String> {
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut decode_limits = image::Limits::default();
    decode_limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(decode_limits);
    let unreadable = |error: image::ImageError, size: Option<(u32, u32)>| match (error, size) {
        (image::ImageError::Limits(_), Some((width, height))) => {
            format!("{}, and {}", why.describe(), too_large(width, height))
        }
        (image::ImageError::Limits(_), None) => {
            format!("{}, and it is too large to shrink", why.describe())
        }
        (error, _) => format!(
            "{}, and its data can't be read to shrink it ({error})",
            why.describe()
        ),
    };
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| unreadable(error, None))?;
    let size = decoder.dimensions();
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut decoded =
        DynamicImage::from_decoder(decoder).map_err(|error| unreadable(error, Some(size)))?;
    decoded.apply_orientation(orientation);
    let lossless = matches!(
        format,
        ImageFormat::Png | ImageFormat::Gif | ImageFormat::Bmp | ImageFormat::Tiff
    );
    let (width, height) = (decoded.width(), decoded.height());
    let mut scale = (f64::from(limits.max_long_edge) / f64::from(width.max(height))).min(1.0);
    if let Some(patches) = limits.max_patches {
        let covered = |scale: f64| patches.patches(scaled(width, scale), scaled(height, scale));
        if covered(scale) > patches.max {
            let side = f64::from(patches.size);
            scale = scale.min(
                (side * side * patches.max as f64 / (f64::from(width) * f64::from(height))).sqrt(),
            );
            while covered(scale) > patches.max {
                scale *= 0.99;
            }
        }
    }
    for _ in 0..8 {
        let resized = if scale < 1.0 {
            let (target_width, target_height) = (scaled(width, scale), scaled(height, scale));
            if scale < 0.5 {
                decoded.thumbnail_exact(target_width, target_height)
            } else {
                decoded.resize_exact(
                    target_width,
                    target_height,
                    image::imageops::FilterType::Triangle,
                )
            }
        } else {
            decoded.clone()
        };
        if lossless
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
    Err(format!("{}, and it can't be shrunk enough", why.describe()))
}

/// `side` scaled by `scale`, at least one pixel.
fn scaled(side: u32, scale: f64) -> u32 {
    ((f64::from(side) * scale).floor() as u32).max(1)
}

/// The length of the base64 of `bytes` bytes.
fn base64_len(bytes: usize) -> usize {
    bytes.div_ceil(3) * 4
}

fn encode_png(image: &DynamicImage) -> Option<Vec<u8>> {
    let mut encoded = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut encoded), ImageFormat::Png)
        .ok()?;
    Some(encoded)
}

/// JPEG at quality 85, transparency laid on white.
fn encode_jpeg(image: &DynamicImage) -> Option<Vec<u8>> {
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

    fn encoded(image: DynamicImage, format: ImageFormat, mime: &str) -> Image {
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), format)
            .unwrap();
        Image::from_bytes(&bytes, mime)
    }

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
        encoded(
            DynamicImage::ImageRgba8(pixels),
            ImageFormat::Png,
            "image/png",
        )
    }

    fn limits(max_image_bytes: usize, max_long_edge: u32) -> ImageLimits {
        ImageLimits {
            max_image_bytes,
            max_long_edge,
            max_images: 100,
            max_request_image_bytes: 100 * MIB,
            ..ANTHROPIC
        }
    }

    fn decoded(image: &Image) -> DynamicImage {
        let Image::Base64 { data, .. } = image else {
            unreachable!()
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data.as_bytes())
            .unwrap();
        image::load_from_memory(&bytes).unwrap()
    }

    fn mime(image: &Image) -> &str {
        match image {
            Image::Base64 { mime_type, .. } => mime_type,
            Image::Url(_) => "url",
        }
    }

    /// An image within the limits is sent as it is, and so is one by URL.
    #[test]
    fn an_image_within_the_limits_is_left_alone() {
        assert_eq!(fit_image(&png(40, 30, false), &limits(MIB, 100)), Ok(None));
        assert_eq!(
            fit_image(&Image::new_url("https://x/y.png"), &limits(1, 1)),
            Ok(None)
        );
    }

    /// Too wide: scaled down to the longest side, keeping its aspect, and
    /// still a PNG.
    #[test]
    fn a_wide_image_is_scaled_to_the_longest_side() {
        let fitted = fit_image(&png(400, 100, false), &limits(MIB, 200))
            .unwrap()
            .unwrap();
        assert_eq!(fitted.dimensions(), Some((200, 50)));
        assert_eq!(mime(&fitted), "image/png");
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

    /// Over a patch limit (OpenAI refuses an image over 30,000 squares of
    /// 32 pixels): scaled down until it covers no more.
    #[test]
    fn an_image_is_fitted_to_a_patch_limit() {
        let limit = PatchLimit { size: 32, max: 100 };
        let few = ImageLimits {
            max_patches: Some(limit),
            ..limits(MIB, 10_000)
        };
        let fitted = fit_image(&png(400, 400, false), &few).unwrap().unwrap();
        let (width, height) = fitted.dimensions().unwrap();
        assert!(limit.patches(width, height) <= 100, "{width}×{height}");
        assert!(width >= 300, "{width}×{height}");
        // The 8K screenshot OpenAI's current models refuse.
        assert!(OPENAI.max_patches.unwrap().patches(7680, 4320) > 30_000);
    }

    /// A JPEG with EXIF orientation 6 (stored sideways, as phones store a
    /// portrait photo).
    fn sideways_photo() -> Image {
        // Stored 40 × 20: red on top, blue below.
        let pixels = image::RgbImage::from_fn(40, 20, |_, y| {
            if y < 10 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            }
        });
        let mut jpeg = Vec::new();
        DynamicImage::ImageRgb8(pixels)
            .write_to(&mut Cursor::new(&mut jpeg), ImageFormat::Jpeg)
            .unwrap();
        // An APP1 segment: "Exif", a big-endian TIFF header, one IFD entry
        // (orientation, a SHORT, 6), no next IFD.
        let mut exif = b"Exif\0\0MM\0\x2a\0\0\0\x08\0\x01".to_vec();
        exif.extend([0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0]);
        let mut with_exif = jpeg[..2].to_vec();
        with_exif.extend([0xff, 0xe1]);
        with_exif.extend(((exif.len() + 2) as u16).to_be_bytes());
        with_exif.extend(exif);
        with_exif.extend(&jpeg[2..]);
        Image::from_bytes(&with_exif, "image/jpeg")
    }

    /// A photo stored sideways with an EXIF orientation is turned upright,
    /// also when it is within the limits.
    #[test]
    fn a_photo_is_turned_upright() {
        let fitted = fit_image(&sideways_photo(), &limits(MIB, 1_000))
            .unwrap()
            .unwrap();
        let upright = decoded(&fitted).to_rgb8();
        assert_eq!(upright.dimensions(), (20, 40));
        // Turned a quarter clockwise, red is on the right.
        assert!(
            upright.get_pixel(17, 20)[0] > 200,
            "{:?}",
            upright.get_pixel(17, 20)
        );
        assert!(
            upright.get_pixel(2, 20)[2] > 200,
            "{:?}",
            upright.get_pixel(2, 20)
        );
    }

    /// A format the provider doesn't take is converted when it can be read:
    /// a BMP to PNG, an animated GIF to its first frame for OpenAI. One that
    /// can't be read here is refused, saying what to do.
    #[test]
    fn formats_the_provider_doesnt_take_are_converted_or_refused() {
        let bmp = encoded(
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(8, 6, image::Rgb([9, 9, 9]))),
            ImageFormat::Bmp,
            "image/bmp",
        );
        let converted = fit_image(&bmp, &ANTHROPIC).unwrap().unwrap();
        assert_eq!(mime(&converted), "image/png");
        assert_eq!(converted.dimensions(), Some((8, 6)));

        let mut gif = Vec::new();
        {
            let mut encoder = image::codecs::gif::GifEncoder::new(&mut gif);
            for color in [[255, 0, 0, 255], [0, 255, 0, 255]] {
                let frame = image::RgbaImage::from_pixel(10, 10, image::Rgba(color));
                encoder.encode_frame(image::Frame::new(frame)).unwrap();
            }
        }
        let animated = Image::from_bytes(&gif, "image/gif");
        assert_eq!(fit_image(&animated, &ANTHROPIC), Ok(None));
        let first = fit_image(&animated, &OPENAI).unwrap().unwrap();
        assert_eq!(mime(&first), "image/png");
        assert!(decoded(&first).to_rgba8().get_pixel(5, 5)[0] > 200);

        let heic = Image::from_bytes(b"\0\0\0\x18ftypheic\0\0\0\0", "image/heic");
        let error = fit_image(&heic, &ANTHROPIC).unwrap_err();
        assert!(
            error.contains("the model doesn't take image/heic images"),
            "{error}"
        );
        assert!(error.contains("Convert it to PNG or JPEG"), "{error}");
        assert_eq!(fit_image(&heic, &GOOGLE), Ok(None));
    }

    /// The CRC-32 of PNG chunks.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// A PNG whose header says 30,000 × 30,000 pixels, in a few bytes.
    fn enormous_png() -> Image {
        let mut ihdr = b"IHDR".to_vec();
        ihdr.extend(30_000u32.to_be_bytes());
        ihdr.extend(30_000u32.to_be_bytes());
        ihdr.extend([8, 6, 0, 0, 0]);
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend(13u32.to_be_bytes());
        bytes.extend(&ihdr);
        bytes.extend(crc32(&ihdr).to_be_bytes());
        bytes.extend(0u32.to_be_bytes());
        bytes.extend(b"IEND");
        bytes.extend(crc32(b"IEND").to_be_bytes());
        Image::from_bytes(&bytes, "image/png")
    }

    /// An image too large to decode is refused at once, saying so, by the
    /// fitting and by the check from its header.
    #[test]
    fn an_image_too_large_to_decode_is_refused_quickly() {
        let started = std::time::Instant::now();
        let error = fit_image(&enormous_png(), &ANTHROPIC).unwrap_err();
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        assert!(error.contains("are too large to shrink"), "{error}");
        assert!(!error.contains("convert it"), "{error}");
        let error = Model::anthropic_claude_opus_5_5()
            .check_image(&enormous_png())
            .unwrap_err();
        assert!(error.contains("too large to shrink"), "{error}");
        assert_eq!(
            Model::anthropic_claude_opus_5_5().check_image(&png(3_000, 10, false)),
            Ok(())
        );
    }

    /// Each provider's limits, OpenRouter's following the model's maker.
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
