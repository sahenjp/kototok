//! Final-vocabulary decoding helpers.

use crate::model::KototokError;

/// Concatenates the byte representations of final token ids.
///
/// Returns `KototokError::InvalidTokenId` for ids outside `final_to_bytes`.
pub fn decode_bytes(final_to_bytes: &[Vec<u8>], ids: &[u32]) -> Result<Vec<u8>, KototokError> {
    let mut total = 0usize;
    for &id in ids {
        match final_to_bytes.get(id as usize) {
            Some(b) => total += b.len(),
            None => return Err(KototokError::InvalidTokenId(id)),
        }
    }
    let mut out = Vec::with_capacity(total);
    for &id in ids {
        out.extend_from_slice(&final_to_bytes[id as usize]);
    }
    Ok(out)
}

/// Decodes final token ids to a `String`; fails on invalid UTF-8 output.
pub fn decode(final_to_bytes: &[Vec<u8>], ids: &[u32]) -> Result<String, KototokError> {
    let bytes = decode_bytes(final_to_bytes, ids)?;
    Ok(String::from_utf8(bytes)?)
}

/// Decodes final token ids lossily, never failing.
/// Unknown ids are skipped.
pub fn decode_lossy(final_to_bytes: &[Vec<u8>], ids: &[u32]) -> String {
    let mut out = Vec::new();
    for &id in ids {
        if let Some(b) = final_to_bytes.get(id as usize) {
            out.extend_from_slice(b);
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
