//! How many input tokens an image takes, by the rule of the model's
//! provider. Hosts and request estimates use this as the images' share of a
//! request.

use crate::message::Image;
use crate::model::{Model, Provider};

/// The tokens an image of unknown size is counted as, outside Anthropic's
/// models (whose limit caps an image's tokens).
const UNKNOWN_SIZE_TOKENS: usize = 1_600;

impl Model {
    /// Counts the input tokens `image` takes in a request to this model using
    /// its provider's rule:
    /// - Anthropic (and Claude Code, and OpenRouter's `anthropic/` models) uses
    ///   a token per 28 × 28 pixels, after scaling the image down to the
    ///   model's limit, 2,576 pixels on the long side and 4,784 tokens for
    ///   Claude 4.7 and later, 1,568 and 1,568 for earlier models.
    /// - OpenAI uses patches of 32 × 32 pixels, after the model's own scaling,
    ///   times its multiplier (1.2 for current models), or tiles of 512
    ///   pixels for older models.
    /// - Google uses 258 tokens for an image up to 384 pixels a side, else 258
    ///   per tile of about two thirds of its shorter side.
    ///
    /// An image whose size can't be read (a URL, a format whose header isn't
    /// read) counts as an Anthropic model's limit, as 2,500 patches on
    /// OpenAI's patch-based models, or as 1,600.
    pub fn image_tokens(&self, image: &Image) -> usize {
        let size = image.dimensions().filter(|(w, h)| *w > 0 && *h > 0);
        let (provider, id) = match (self.provider.clone(), self.id.split_once('/')) {
            (Provider::OpenRouter, Some(("anthropic", id))) => (Provider::Anthropic, id),
            (Provider::OpenRouter, Some(("openai", id))) => (Provider::OpenAI, id),
            (Provider::OpenRouter, Some(("google", id))) => (Provider::Google, id),
            (provider, _) => (provider, self.id.as_str()),
        };
        match provider {
            Provider::Anthropic | Provider::ClaudeCode => {
                anthropic_tokens(size, AnthropicTier::of(id))
            }
            Provider::OpenAI => openai_tokens(size, id),
            Provider::Google => size.map_or(UNKNOWN_SIZE_TOKENS, gemini_tokens),
            Provider::OpenRouter => anthropic_tokens(size, AnthropicTier::STANDARD),
        }
    }
}

/// An Anthropic model's image limit: the long side and the tokens an
/// image is scaled down to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AnthropicTier {
    long_edge: u32,
    tokens: usize,
}

impl AnthropicTier {
    /// Claude 4.7 and later.
    pub(crate) const HIGH_RESOLUTION: Self = Self {
        long_edge: 2_576,
        tokens: 4_784,
    };
    /// Earlier models.
    pub(crate) const STANDARD: Self = Self {
        long_edge: 1_568,
        tokens: 1_568,
    };

    /// The tier of Claude model `id`: from its version, the first two
    /// numbers of its ID (`claude-opus-4-7`, `claude-sonnet-4.5`,
    /// `claude-3-5-sonnet`).
    pub(crate) fn of(id: &str) -> Self {
        let mut numbers = id
            .split(['-', '.', '/', '@', ':'])
            .filter_map(|part| part.parse::<u32>().ok());
        match (numbers.next(), numbers.next()) {
            (Some(major), _) if major >= 5 => Self::HIGH_RESOLUTION,
            (Some(4), Some(minor)) if (7..100).contains(&minor) => Self::HIGH_RESOLUTION,
            _ => Self::STANDARD,
        }
    }
}

/// Anthropic's rule: `⌈w / 28⌉ × ⌈h / 28⌉`, after scaling to the tier's
/// long side and tokens.
pub(crate) fn anthropic_tokens(size: Option<(u32, u32)>, tier: AnthropicTier) -> usize {
    let Some((width, height)) = size else {
        return tier.tokens;
    };
    let tokens = |scale: f64| {
        let side = |pixels: u32| ((f64::from(pixels) * scale).floor().max(1.0) / 28.0).ceil();
        (side(width) * side(height)) as usize
    };
    let scale = (f64::from(tier.long_edge) / f64::from(width.max(height))).min(1.0);
    if tokens(scale) <= tier.tokens {
        return tokens(scale);
    }
    // The largest scale within the tokens.
    let (mut within, mut over) = (0.0, scale);
    for _ in 0..40 {
        let middle = (within + over) / 2.0;
        if tokens(middle) <= tier.tokens {
            within = middle;
        } else {
            over = middle;
        }
    }
    tokens(within)
}

/// OpenAI's rule for model `id`.
fn openai_tokens(size: Option<(u32, u32)>, id: &str) -> usize {
    // Tile-based models: base tokens, and tokens per 512-pixel tile.
    let tiles = if id.starts_with("gpt-4o-mini") {
        Some((2_833, 5_667))
    } else if id.starts_with("gpt-4o")
        || (id.starts_with("gpt-4.1")
            && !id.starts_with("gpt-4.1-mini")
            && !id.starts_with("gpt-4.1-nano"))
    {
        Some((85, 170))
    } else if id.starts_with("gpt-5.1") || id == "gpt-5" || id.starts_with("gpt-5-20") {
        Some((70, 140))
    } else if id.starts_with('o') && !id.starts_with("o4-mini") {
        Some((75, 150))
    } else {
        None
    };
    if let Some((base, tile)) = tiles {
        let Some((width, height)) = size else {
            return UNKNOWN_SIZE_TOKENS;
        };
        // Within 2,048 pixels a side, then the shorter side within 768.
        let fit = (2_048.0 / f64::from(width.max(height))).min(1.0);
        let (mut w, mut h) = (
            (f64::from(width) * fit).floor(),
            (f64::from(height) * fit).floor(),
        );
        if w.min(h) > 768.0 {
            let shorter = 768.0 / w.min(h);
            (w, h) = ((w * shorter).floor(), (h * shorter).floor());
        }
        return base + tile * ((w / 512.0).ceil() * (h / 512.0).ceil()) as usize;
    }
    // Patch-based models: the longest side and the patch budget the
    // default detail scales an image to, and the multiplier.
    let (long_edge, budget, multiplier) = if id.starts_with("gpt-6") || id.starts_with("gpt-5.6") {
        (65_535, None, 1.2)
    } else if id.starts_with("gpt-5.5") {
        (6_000, Some(10_000), 1.2)
    } else if id.starts_with("gpt-5.4") {
        (2_048, Some(2_500), 1.2)
    } else if id.starts_with("gpt-4.1-mini") {
        (2_048, Some(6_144), 1.62)
    } else if id.starts_with("gpt-5-nano") {
        (2_048, Some(6_144), 1.5)
    } else if id.starts_with("o4-mini") {
        (2_048, Some(6_144), 1.72)
    } else {
        (2_048, Some(6_144), 1.2)
    };
    let Some((width, height)) = size else {
        return (budget.unwrap_or(2_500) as f64 * multiplier).ceil() as usize;
    };
    let patches = |w: f64, h: f64| ((w / 32.0).ceil() * (h / 32.0).ceil()) as usize;
    let fit = (f64::from(long_edge) / f64::from(width.max(height))).min(1.0);
    let (mut w, mut h) = (
        (f64::from(width) * fit).floor().max(1.0),
        (f64::from(height) * fit).floor().max(1.0),
    );
    if let Some(budget) = budget
        && patches(w, h) > budget
    {
        let shrink = (1_024.0 * budget as f64 / (w * h)).sqrt();
        let adjusted = shrink
            * ((w * shrink / 32.0).floor() / (w * shrink / 32.0))
                .min((h * shrink / 32.0).floor() / (h * shrink / 32.0));
        w = (w * adjusted).floor().max(1.0);
        h = (h * adjusted).floor().max(1.0);
    }
    (patches(w, h) as f64 * multiplier).ceil() as usize
}

/// Google's rule: 258 tokens up to 384 pixels a side, else 258 per tile,
/// tiles of about two thirds of the shorter side.
fn gemini_tokens((width, height): (u32, u32)) -> usize {
    if width <= 384 && height <= 384 {
        return 258;
    }
    let crop = (f64::from(width.min(height)) / 1.5).floor().max(1.0);
    let tiles = (f64::from(width) / crop).ceil() * (f64::from(height) / crop).ceil();
    258 * tiles as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Image {
        let mut bytes = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        bytes.extend(width.to_be_bytes());
        bytes.extend(height.to_be_bytes());
        Image::from_bytes(&bytes, "image/png")
    }

    /// Anthropic's own table of image sizes and tokens, for both tiers.
    #[test]
    fn anthropic_images_count_as_its_table_shows() {
        let opus = Model::anthropic_claude_opus_5_5();
        let mut older = opus.clone();
        older.id = "claude-sonnet-4-5".into();
        for (size, standard, high) in [
            ((200, 200), 64, 64),
            ((1000, 1000), 1296, 1296),
            ((1092, 1092), 1521, 1521),
            ((1920, 1080), 1560, 2691),
            ((2000, 1500), 1564, 3888),
            ((3840, 2160), 1560, 4784),
        ] {
            let image = png(size.0, size.1);
            assert_eq!(older.image_tokens(&image), standard, "{size:?} standard");
            assert_eq!(opus.image_tokens(&image), high, "{size:?} high resolution");
        }
        assert_eq!(
            AnthropicTier::of("claude-opus-4-7"),
            AnthropicTier::HIGH_RESOLUTION
        );
        assert_eq!(
            AnthropicTier::of("claude-3-5-sonnet-20241022"),
            AnthropicTier::STANDARD
        );
        assert_eq!(
            AnthropicTier::of("claude-sonnet-4.5"),
            AnthropicTier::STANDARD
        );
        assert_eq!(
            Model::openrouter("anthropic/claude-sonnet-5").image_tokens(&png(2000, 1500)),
            3888
        );
        assert_eq!(opus.image_tokens(&Image::new_url("https://x/y.png")), 4_784);
    }

    /// OpenAI's examples: patches of 32 pixels times 1.2 on current models,
    /// within the model's own budget.
    #[test]
    fn openai_images_count_by_patches_or_tiles() {
        let luna = Model::openai_gpt_5_6_luna();
        assert_eq!(luna.image_tokens(&png(1024, 1024)), 1229);
        assert_eq!(luna.image_tokens(&png(4096, 512)), 2458);
        // The original size: an 8K screenshot is 32,400 patches.
        assert_eq!(luna.image_tokens(&png(7680, 4320)), 38_880);
        let mut older = luna.clone();
        older.id = "gpt-5.4".into();
        // Within 2,500 patches: 1600 × 1600.
        assert_eq!(older.image_tokens(&png(2048, 2048)), 3000);
        older.id = "gpt-4o".into();
        // 2048 × 4096 → 1024 × 2048 → 768 × 1536: 2 × 3 tiles.
        assert_eq!(older.image_tokens(&png(2048, 4096)), 85 + 170 * 6);
    }

    /// Google's: one tile up to 384 pixels a side, else tiles of two thirds
    /// of the shorter side.
    #[test]
    fn gemini_images_count_by_tiles() {
        let gemini = Model::get(Provider::Google, "gemini-3.5-flash").unwrap();
        assert_eq!(gemini.image_tokens(&png(300, 200)), 258);
        assert_eq!(gemini.image_tokens(&png(960, 540)), 258 * 6);
    }
}
