//! Internal vocabulary initialization.
//!
//! Layout: ids `0..128` are `Ascii` (mergeable), ids `128..256` are
//! `ByteFallback` (mergeable only under `ByteMode::RawBytes`), then `Special`
//! tokens, then `AtomicChar` entries (only under `ByteMode::Utf8Atomic`).
//! Atomic chars are chosen by descending corpus frequency, tie-broken by
//! codepoint, and capped so `256 + specials + atomic_chars <= vocab_size`;
//! rare chars not selected still encode via `ByteFallback` bytes.

use kototok_core::{ByteMode, InternalToken, TokenKind};
use rustc_hash::FxHashMap;

use crate::config::TrainConfig;
use crate::trainer::TrainError;

/// Builds the initial internal token table plus the `char -> internal_id`
/// map used for `Utf8Atomic` initial ids.
pub fn init_tokens(
    cfg: &TrainConfig,
    char_counts: &FxHashMap<char, u64>,
) -> Result<(Vec<InternalToken>, FxHashMap<char, u32>), TrainError> {
    if cfg.vocab_size < 256 + cfg.special_tokens.len() {
        return Err(TrainError::VocabTooSmall(cfg.vocab_size));
    }

    let mut tokens: Vec<InternalToken> = Vec::with_capacity(cfg.vocab_size.max(256));
    for b in 0..256u32 {
        tokens.push(InternalToken {
            internal_id: b,
            bytes: vec![b as u8],
            kind: if b < 128 {
                TokenKind::Ascii
            } else {
                TokenKind::ByteFallback
            },
            final_id: Some(b),
        });
    }
    for s in &cfg.special_tokens {
        let internal_id = tokens.len() as u32;
        tokens.push(InternalToken {
            internal_id,
            bytes: s.clone().into_bytes(),
            kind: TokenKind::Special,
            final_id: Some(internal_id),
        });
    }

    let mut char_to_id = FxHashMap::default();
    if cfg.pretok.byte_mode == ByteMode::Utf8Atomic {
        let budget = cfg.vocab_size - tokens.len();
        let mut candidates: Vec<(u64, char)> = char_counts
            .iter()
            .filter(|(ch, n)| !ch.is_ascii() && **n >= cfg.min_char_freq)
            .map(|(&ch, &n)| (n, ch))
            .collect();
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let cap = cfg.max_atomic_chars.map_or(budget, |max| max.min(budget));
        candidates.truncate(cap);
        for (_, ch) in candidates {
            let internal_id = tokens.len() as u32;
            let mut buf = [0u8; 4];
            tokens.push(InternalToken {
                internal_id,
                bytes: ch.encode_utf8(&mut buf).as_bytes().to_vec(),
                kind: TokenKind::AtomicChar,
                final_id: Some(internal_id),
            });
            char_to_id.insert(ch, internal_id);
        }
    }

    Ok((tokens, char_to_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_atomic_init_caps_atomic_chars() {
        // vocab_size 260 leaves room for only 4 atomic chars.
        let cfg = TrainConfig {
            vocab_size: 260,
            ..TrainConfig::default()
        };
        let mut counts = FxHashMap::default();
        counts.insert('あ', 10u64);
        counts.insert('い', 20u64);
        counts.insert('う', 30u64);
        counts.insert('え', 40u64);
        counts.insert('お', 50u64);
        let (tokens, map) = init_tokens(&cfg, &counts).unwrap();
        assert_eq!(tokens.len(), 260);
        assert_eq!(map.len(), 4);
        // top-4 by frequency: お え う い (あ dropped)
        assert!(map.contains_key(&'お') && !map.contains_key(&'あ'));
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(t.internal_id as usize, i);
        }
    }

    #[test]
    fn max_atomic_chars_caps_candidates() {
        let cfg = TrainConfig {
            vocab_size: 300,
            max_atomic_chars: Some(2),
            ..TrainConfig::default()
        };
        let mut counts = FxHashMap::default();
        for ch in ['あ', 'い', 'う', 'え'] {
            counts.insert(ch, 10u64);
        }
        let (_, map) = init_tokens(&cfg, &counts).unwrap();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn raw_bytes_has_no_atomic_chars() {
        let mut cfg = TrainConfig::default();
        cfg.pretok.byte_mode = ByteMode::RawBytes;
        let mut counts = FxHashMap::default();
        counts.insert('あ', 100u64);
        let (tokens, map) = init_tokens(&cfg, &counts).unwrap();
        assert_eq!(tokens.len(), 256);
        assert!(map.is_empty());
        assert_eq!(tokens[255].kind, TokenKind::ByteFallback);
    }

    #[test]
    fn vocab_too_small_errors() {
        let cfg = TrainConfig {
            vocab_size: 200,
            ..TrainConfig::default()
        };
        assert!(init_tokens(&cfg, &FxHashMap::default()).is_err());
    }
}
