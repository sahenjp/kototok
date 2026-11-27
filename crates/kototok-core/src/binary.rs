//! Model serialization: `.ktok` binary container and JSON.
//!
//! Binary layout (little-endian):
//! ```text
//! [0..4]   KTOK_MAGIC          "KTOK"
//! [4..8]   u32                 format version (KTOK_FORMAT_VERSION)
//! [8..16]  u64                 payload length in bytes
//! [16..]   payload             serde_json encoding of `Model`
//! ```
//! All load paths validate lengths before slicing — truncated or corrupt
//! input produces `KototokError::InvalidFormat`, never a panic.

use std::path::Path;

use crate::model::{KTOK_FORMAT_VERSION, KTOK_MAGIC, KototokError, Model};

const HEADER_LEN: usize = 4 + 4 + 8;

fn invalid(msg: impl Into<String>) -> KototokError {
    KototokError::InvalidFormat(msg.into())
}

impl Model {
    /// Serializes this model to the `.ktok` binary container.
    pub fn save_binary(&self, path: impl AsRef<Path>) -> Result<(), KototokError> {
        let payload = serde_json::to_vec(self)?;
        let mut buf = Vec::with_capacity(HEADER_LEN + payload.len());
        buf.extend_from_slice(&KTOK_MAGIC);
        buf.extend_from_slice(&KTOK_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        buf.extend_from_slice(&payload);
        std::fs::write(path, buf)?;
        Ok(())
    }

    /// Loads a `.ktok` binary container. Errors (never panics) on short,
    /// truncated, or malformed input.
    pub fn load_binary(path: impl AsRef<Path>) -> Result<Self, KototokError> {
        let bytes = std::fs::read(path)?;
        Self::from_binary_bytes(&bytes)
    }

    /// Parses a `.ktok` container from raw bytes.
    pub fn from_binary_bytes(bytes: &[u8]) -> Result<Self, KototokError> {
        if bytes.len() < HEADER_LEN {
            return Err(invalid("file too short for KTOK header"));
        }
        if bytes[0..4] != KTOK_MAGIC {
            return Err(invalid("missing KTOK magic header"));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != KTOK_FORMAT_VERSION {
            return Err(invalid(format!(
                "unsupported KTOK format version {version} (expected {KTOK_FORMAT_VERSION})"
            )));
        }
        let payload_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let rest = &bytes[HEADER_LEN..];
        if rest.len() < payload_len {
            return Err(invalid("truncated KTOK payload"));
        }
        if rest.len() > payload_len {
            return Err(invalid("trailing bytes after KTOK payload"));
        }
        let model: Model = serde_json::from_slice(&rest[..payload_len])?;
        model.validate_final_id_sparsity()?;
        Ok(model)
    }

    /// Serializes this model as pretty-printed JSON.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), KototokError> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(file), self)?;
        Ok(())
    }

    /// Loads a model from a JSON file.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, KototokError> {
        let bytes = std::fs::read(path)?;
        let model: Model = serde_json::from_slice(&bytes)?;
        model.validate_final_id_sparsity()?;
        Ok(model)
    }

    /// Loads a model, detecting its native or interoperable format.
    pub fn load_auto(path: impl AsRef<Path>) -> Result<Self, KototokError> {
        crate::formats::load_auto(path.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{InternalToken, ModelMetadata, PreTokenizerConfig, TokenKind, VocabEvent};

    fn toy_model() -> Model {
        Model {
            metadata: ModelMetadata {
                name: "toy".into(),
                version: 1,
                vocab_size: 256,
                phase1_vocab_size: 256,
                phase2_start_rank: 0,
                pruned_count: 0,
                invalid_utf8_learned_tokens: 0,
                train_notes: "test".into(),
            },
            pretok: PreTokenizerConfig::default(),
            special_tokens: vec![("<|eos|>".into(), 999)],
            internal_tokens: (0..256u32)
                .map(|b| InternalToken {
                    internal_id: b,
                    bytes: vec![b as u8],
                    kind: if b < 128 {
                        TokenKind::Ascii
                    } else {
                        TokenKind::ByteFallback
                    },
                    final_id: Some(b),
                })
                .collect(),
            events: vec![VocabEvent::Merge {
                left: 97,
                right: 98,
                result: 256,
            }],
            tag_ranges: Vec::new(),
            tagged_prunes: Vec::new(),
        }
    }

    #[test]
    fn binary_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("kototok_test_roundtrip.ktok");
        let model = toy_model();
        model.save_binary(&path).unwrap();
        let loaded = Model::load_auto(&path).unwrap();
        assert_eq!(model, loaded);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn json_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("kototok_test_roundtrip.json");
        let model = toy_model();
        model.save_json(&path).unwrap();
        let loaded = Model::load_auto(&path).unwrap();
        assert_eq!(model, loaded);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sparse_final_id_rejected_without_allocation() {
        let mut model = toy_model();
        model.internal_tokens[0].final_id = Some(u32::MAX - 1);
        assert!(matches!(
            model.validate_final_id_sparsity(),
            Err(KototokError::InvalidFormat(_))
        ));
        let payload = serde_json::to_vec(&model).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&KTOK_MAGIC);
        bytes.extend_from_slice(&KTOK_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        assert!(matches!(
            Model::from_binary_bytes(&bytes),
            Err(KototokError::InvalidFormat(_))
        ));
    }

    #[test]
    fn corrupt_binary_errors_no_panic() {
        // Truncated header
        assert!(Model::from_binary_bytes(b"KT").is_err());
        // Bad magic
        assert!(Model::from_binary_bytes(b"NOPE............").is_err());
        // Bad version
        let mut bad_version = Vec::new();
        bad_version.extend_from_slice(&KTOK_MAGIC);
        bad_version.extend_from_slice(&999u32.to_le_bytes());
        bad_version.extend_from_slice(&0u64.to_le_bytes());
        assert!(Model::from_binary_bytes(&bad_version).is_err());
        // Declared payload longer than actual
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&KTOK_MAGIC);
        truncated.extend_from_slice(&KTOK_FORMAT_VERSION.to_le_bytes());
        truncated.extend_from_slice(&100u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        assert!(Model::from_binary_bytes(&truncated).is_err());
        // Trailing garbage
        let model = toy_model();
        let payload = serde_json::to_vec(&model).unwrap();
        let mut trailing = Vec::new();
        trailing.extend_from_slice(&KTOK_MAGIC);
        trailing.extend_from_slice(&KTOK_FORMAT_VERSION.to_le_bytes());
        trailing.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        trailing.extend_from_slice(&payload);
        trailing.extend_from_slice(b"junk");
        assert!(Model::from_binary_bytes(&trailing).is_err());
        // Valid header, corrupt payload
        let mut bad_payload = Vec::new();
        bad_payload.extend_from_slice(&KTOK_MAGIC);
        bad_payload.extend_from_slice(&KTOK_FORMAT_VERSION.to_le_bytes());
        bad_payload.extend_from_slice(&3u64.to_le_bytes());
        bad_payload.extend_from_slice(b"!!!");
        assert!(Model::from_binary_bytes(&bad_payload).is_err());
    }
}
