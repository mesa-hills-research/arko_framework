//! Content hashing shared by durable steps and verification baselines.

use std::fmt::Write;

use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of `bytes`. Durable journals persist these hashes,
/// so the format must not change.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    #[test]
    fn sha256_hex_is_lowercase_hex_of_the_digest() {
        assert_eq!(
            super::sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            super::sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
