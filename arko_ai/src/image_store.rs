//! Images kept outside the JSON they belong to. A host that saves
//! conversations often (a task journal, checkpoints) saves each image once,
//! in a store of its own, and writes a reference in its place. Inside
//! [`with_image_store`], an [`Image::Base64`] serializes as
//! `{"Base64": {"mime_type": .., "stored": REFERENCE}}`, and such JSON reads
//! back through the store. Outside, images serialize with their data, as
//! always, and JSON written with data reads back without a store.
//!
//! An image the store can't give back (its file was deleted, or is damaged)
//! reads as a placeholder ([`Image::unavailable`]) with the reason, so the
//! rest of the value still loads. arko_agents' tasks turn placeholders into
//! a line of text saying the image is gone ([`MISSING_IMAGE_TYPE`]).

use std::cell::RefCell;
use std::sync::Arc;

use serde::de::Error as _;
use serde::ser::{Error as _, SerializeStructVariant};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::message::Image;

/// Where images are saved while a value is serialized, and read from while
/// it is deserialized. See [`with_image_store`].
pub trait ImageStore: Send + Sync {
    /// Saves the image (base64 `data`, of `mime_type`) and returns the
    /// reference written in its place. Saving the same image again returns
    /// the same reference.
    fn save(&self, data: &str, mime_type: &str) -> Result<String, String>;

    /// The base64 data of the image saved under `reference`, of
    /// `mime_type`.
    fn load(&self, reference: &str, mime_type: &str) -> Result<String, String>;
}

thread_local! {
    static STORE: RefCell<Option<Arc<dyn ImageStore>>> = const { RefCell::new(None) };
}

/// Runs `f` with `store` as the thread's image store. Images serialized in
/// `f` are saved in it and written as references, and references read in
/// `f` are loaded from it. The previous store is back afterwards.
pub fn with_image_store<R>(store: Arc<dyn ImageStore>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Arc<dyn ImageStore>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            STORE.with(|store| *store.borrow_mut() = previous);
        }
    }
    let previous = STORE.with(|current| current.borrow_mut().replace(store));
    let _restore = Restore(previous);
    f()
}

/// A hash of `data`'s length and of 32 slices of it spread evenly over it
/// (all of it when short). A cheap stand-in for an image's content, to find
/// candidates that are then compared in full, or where an image only needs
/// telling apart from the images around it.
pub fn sample_hash(data: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    const SLICE: usize = 64;
    const SLICES: usize = 32;
    let bytes = data.as_bytes();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.len().hash(&mut hasher);
    if bytes.len() <= SLICE * SLICES {
        hasher.write(bytes);
    } else {
        let last = bytes.len() - SLICE;
        for index in 0..SLICES {
            let start = last * index / (SLICES - 1);
            hasher.write(&bytes[start..start + SLICE]);
        }
    }
    hasher.finish()
}

/// The MIME type of a placeholder for an image a store couldn't give back.
/// Its data is the reason.
pub const MISSING_IMAGE_TYPE: &str = "application/x-missing-image";

impl Image {
    /// A placeholder for an image that is gone, saying why.
    pub fn unavailable(reason: &str) -> Self {
        Image::Base64 {
            data: reason.into(),
            mime_type: MISSING_IMAGE_TYPE.to_string(),
        }
    }

    /// Why the image is gone, for a placeholder ([`Image::unavailable`]).
    pub fn unavailable_reason(&self) -> Option<&str> {
        match self {
            Image::Base64 { data, mime_type } if mime_type == MISSING_IMAGE_TYPE => Some(data),
            _ => None,
        }
    }
}

fn current() -> Option<Arc<dyn ImageStore>> {
    STORE.with(|store| store.borrow().clone())
}

impl Serialize for Image {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Image::Url(url) => serializer.serialize_newtype_variant("Image", 0, "Url", url),
            Image::Base64 { data, mime_type } => {
                let stored = match current() {
                    Some(store) => Some(store.save(data, mime_type).map_err(S::Error::custom)?),
                    None => None,
                };
                let mut variant = serializer.serialize_struct_variant("Image", 1, "Base64", 2)?;
                match &stored {
                    Some(reference) => variant.serialize_field("stored", reference)?,
                    None => variant.serialize_field("data", &**data)?,
                }
                variant.serialize_field("mime_type", mime_type)?;
                variant.end()
            }
        }
    }
}

/// How an image is written: with its data, or with the reference of a store
/// that holds it.
#[derive(Deserialize)]
enum Repr {
    Url(String),
    Base64 {
        #[serde(default)]
        data: Option<String>,
        mime_type: String,
        #[serde(default)]
        stored: Option<String>,
    },
}

impl<'de> Deserialize<'de> for Image {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match Repr::deserialize(deserializer)? {
            Repr::Url(url) => Image::Url(url),
            Repr::Base64 {
                data: Some(data),
                mime_type,
                ..
            } => Image::Base64 {
                data: data.into(),
                mime_type,
            },
            Repr::Base64 {
                data: None,
                mime_type,
                stored: Some(reference),
            } => {
                let store = current().ok_or_else(|| {
                    D::Error::custom(format!(
                        "image '{reference}' is saved outside this JSON, and no image store is set to read it"
                    ))
                })?;
                match store.load(&reference, &mime_type) {
                    Ok(data) => Image::Base64 {
                        data: data.into(),
                        mime_type,
                    },
                    Err(reason) => {
                        log::warn!("An image can't be read back: {reason}");
                        Image::unavailable(&reason)
                    }
                }
            }
            Repr::Base64 {
                data: None,
                stored: None,
                ..
            } => {
                return Err(D::Error::custom(
                    "an image has neither data nor a stored reference",
                ));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Memory(Mutex<HashMap<String, String>>);

    impl ImageStore for Memory {
        fn save(&self, data: &str, _mime_type: &str) -> Result<String, String> {
            let reference = format!("ref-{}", data.len());
            self.0
                .lock()
                .unwrap()
                .insert(reference.clone(), data.to_string());
            Ok(reference)
        }
        fn load(&self, reference: &str, _mime_type: &str) -> Result<String, String> {
            self.0
                .lock()
                .unwrap()
                .get(reference)
                .cloned()
                .ok_or_else(|| format!("no image {reference}"))
        }
    }

    /// Without a store an image is written with its data, as always. Inside
    /// one, it is written as a reference, which reads back through the store
    /// only.
    #[test]
    fn a_store_holds_the_data_and_the_json_a_reference() {
        let image = Image::new_base64("cGl4ZWxz", "image/png");
        let inline = serde_json::to_value(&image).unwrap();
        assert_eq!(
            inline,
            serde_json::json!({"Base64": {"data": "cGl4ZWxz", "mime_type": "image/png"}})
        );
        assert_eq!(serde_json::from_value::<Image>(inline).unwrap(), image);

        let store: Arc<dyn ImageStore> = Arc::new(Memory::default());
        let stored = with_image_store(store.clone(), || serde_json::to_value(&image).unwrap());
        assert_eq!(
            stored,
            serde_json::json!({"Base64": {"stored": "ref-8", "mime_type": "image/png"}})
        );
        let error = serde_json::from_value::<Image>(stored.clone()).unwrap_err();
        assert!(error.to_string().contains("no image store"), "{error}");
        let read = with_image_store(store.clone(), || {
            serde_json::from_value::<Image>(stored).unwrap()
        });
        assert_eq!(read, image);
        // An image the store can't give back reads as a placeholder saying
        // why, so what holds it still loads.
        let gone = serde_json::json!({"Base64": {"stored": "ref-99", "mime_type": "image/png"}});
        let placeholder =
            with_image_store(store, || serde_json::from_value::<Image>(gone).unwrap());
        assert_eq!(placeholder.unavailable_reason(), Some("no image ref-99"));
        let url = Image::new_url("https://example.com/a.png");
        assert_eq!(
            serde_json::to_value(&url).unwrap(),
            serde_json::json!({"Url": "https://example.com/a.png"})
        );
    }
}
