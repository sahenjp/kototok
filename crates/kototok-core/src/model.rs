use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Binary magic header for `.ktok` serialized tokenizer models.
pub const KTOK_MAGIC: [u8; 4] = *b"KTOK";

/// Binary and metadata format version.
pub const KTOK_FORMAT_VERSION: u32 = 1;

/// Pre-tokenization strategy controlling boundary splits before BPE merges.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizerMode {
    /// Asymmetric Japanese script-transition DFA: allows `Kanji -> Hiragana`
    /// okurigana inside a chunk while splitting `Hiragana -> Kanji` and other
    /// script transitions.
    AsymScript,
    /// Strict script-boundary baseline: splits on every script transition,
    /// including `Kanji | Hiragana`.
    StrictScript,
    /// GPT-style pre-tokenization baseline: groups all Unicode letters
    /// together regardless of CJK script boundaries.
    StandardGpt,
}

/// Byte representation strategy during vocabulary initialization and merging.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteMode {
    /// IDs `0..128` are mergeable ASCII, IDs `128..256` are non-mergeable
    /// `ByteFallback`, and IDs `>= 256` are complete valid UTF-8 characters
    /// or merged tokens.
    Utf8Atomic,
    /// Standard BBPE baseline where all `0..256` raw bytes are mergeable.
    RawBytes,
}

/// Configuration parameters for DFA pre-tokenization and Phase 2 segments.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PreTokenizerConfig {
    pub mode: PreTokenizerMode,
    pub byte_mode: ByteMode,
    pub attach_ascii_leading_space: bool,
    pub max_digit_run: usize,
    pub max_chunk_chars: usize,
    pub max_segment_chunks: usize,
}

impl Default for PreTokenizerConfig {
    fn default() -> Self {
        Self {
            mode: PreTokenizerMode::AsymScript,
            byte_mode: ByteMode::Utf8Atomic,
            attach_ascii_leading_space: true,
            max_digit_run: 3,
            max_chunk_chars: 128,
            max_segment_chunks: 64,
        }
    }
}

/// Semantic origin of an internal token entry.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Ascii,
    ByteFallback,
    Special,
    AtomicChar,
    MergedPhase1,
    MergedPhase2,
}

/// Entry in the internal token table, indexed by `internal_id`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct InternalToken {
    pub internal_id: u32,
    pub bytes: Vec<u8>,
    pub kind: TokenKind,
    /// `Some(0..vocab_size)` if active in the exported vocabulary, `None` if pruned.
    pub final_id: Option<u32>,
}

/// Chronological vocabulary mutation event recorded during training.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum VocabEvent {
    Merge { left: u32, right: u32, result: u32 },
    Prune { target: u32, split_into: Vec<u32> },
}

/// Metadata summarizing a trained `kototok` model.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ModelMetadata {
    pub name: String,
    pub version: u32,
    pub vocab_size: usize,
    pub phase1_vocab_size: usize,
    pub phase2_start_rank: u32,
    pub pruned_count: usize,
    pub invalid_utf8_learned_tokens: usize,
    pub train_notes: String,
}

/// Script composition breakdown across active vocabulary tokens.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct ScriptBreakdown {
    pub ascii: usize,
    pub hiragana: usize,
    pub katakana: usize,
    pub kanji: usize,
    pub kanji_hiragana: usize,
    pub mixed_utf8: usize,
    pub byte_fallback: usize,
    pub invalid_utf8: usize,
}

/// UTF-8 validity and vocabulary composition audit report.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Utf8AuditReport {
    pub total_internal_tokens: usize,
    pub alive_vocab_size: usize,
    pub alive_non_fallback_tokens: usize,
    pub byte_fallback_tokens: usize,
    pub pruned_tokens: usize,
    pub atomic_char_alive: usize,
    pub phase1_merged_alive: usize,
    pub phase2_merged_alive: usize,
    pub valid_utf8_learned_tokens: usize,
    pub invalid_utf8_learned_tokens: usize,
    pub invalid_utf8_rate: f64,
    pub script_breakdown: ScriptBreakdown,
}

/// A contiguous range of `events` carrying a provenance tag. Grafted
/// merge layers (e.g. `okuri`, `kozh`, `casual`) record their event span
/// so a load-time policy can enable/disable whole layers — one `.ktok`
/// file serves multiple deterministic segmentation policies.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct TagRange {
    pub start: u32,
    pub end: u32,
    pub tag: String,
}

/// Complete serialized representation of a trained `kototok` tokenizer.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub metadata: ModelMetadata,
    pub pretok: PreTokenizerConfig,
    pub special_tokens: Vec<(String, u32)>,
    pub internal_tokens: Vec<InternalToken>,
    pub events: Vec<VocabEvent>,
    /// Provenance-tagged event ranges (appended graft layers). Empty for
    /// models produced before tagging existed.
    #[serde(default)]
    pub tag_ranges: Vec<TagRange>,
    /// Sparse tag assignments for `Prune` events, keyed by target
    /// internal id. Prunes are rank-sensitive and must stay mid-list,
    /// so they cannot live inside a `tag_ranges` span; this sparse map
    /// makes them toggleable without moving them.
    #[serde(default)]
    pub tagged_prunes: Vec<(u32, String)>,
}

impl Model {
    /// Merge events inside a `tag_ranges` span whose tag is disabled are
    /// dropped; the produced tokens simply become unreachable (graceful
    /// degradation to smaller pieces). `Prune` events always stay — they
    /// carry boundary corrections independent of merge layers.
    pub fn filtered_by_tags(&self, enabled: &std::collections::HashSet<String>) -> Model {
        let mut m = self.clone();
        let disabled: Vec<&TagRange> = m
            .tag_ranges
            .iter()
            .filter(|r| !enabled.contains(&r.tag))
            .collect();
        let disabled_prunes: std::collections::HashSet<u32> = m
            .tagged_prunes
            .iter()
            .filter(|(_, t)| !enabled.contains(t))
            .map(|(id, _)| *id)
            .collect();
        if disabled.is_empty() && disabled_prunes.is_empty() {
            return m;
        }
        m.events = m
            .events
            .iter()
            .enumerate()
            .filter(|(i, ev)| {
                let i = *i as u32;
                if disabled.iter().any(|r| r.start <= i && i < r.end) {
                    return false;
                }
                if let VocabEvent::Prune { target, .. } = ev
                    && disabled_prunes.contains(target)
                {
                    return false;
                }
                true
            })
            .map(|(_, ev)| ev.clone())
            .collect();
        m
    }

    /// All tag names present in the model.
    pub fn tags(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tag_ranges.iter().map(|r| r.tag.clone()).collect();
        v.extend(self.tagged_prunes.iter().map(|(_, t)| t.clone()));
        v.sort();
        v.dedup();
        v
    }
}

/// Errors returned by `kototok-core` encoding, decoding, and serialization APIs.
#[derive(Error, Debug)]
pub enum KototokError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid model format: {0}")]
    InvalidFormat(String),
    #[error("invalid token id: {0}")]
    InvalidTokenId(u32),
    #[error("decoded bytes are not valid UTF-8: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("pruned internal token {0} survived encoding")]
    PrunedTokenEmitted(u32),
}

impl Model {
    /// Returns the number of active tokens in the final vocabulary.
    pub fn vocab_size(&self) -> usize {
        let count = self
            .internal_tokens
            .iter()
            .filter(|tok| tok.final_id.is_some())
            .count();
        if count > 0 {
            count
        } else {
            self.metadata.vocab_size
        }
    }

    /// Maps each `final_id` (`0..vocab_size()`) to its byte slice.
    pub fn final_id_to_bytes(&self) -> Vec<&[u8]> {
        let size = self
            .internal_tokens
            .iter()
            .filter_map(|tok| tok.final_id)
            .map(|id| id as usize + 1)
            .max()
            .unwrap_or(0);
        let mut table: Vec<&[u8]> = vec![&[]; size];
        for tok in &self.internal_tokens {
            if let Some(fid) = tok.final_id {
                table[fid as usize] = tok.bytes.as_slice();
            }
        }
        table
    }

    /// Audits UTF-8 validity, pruning counts, and script breakdown across the vocabulary.
    pub fn audit_utf8_validity(&self) -> Utf8AuditReport {
        let mut alive_vocab_size = 0usize;
        let mut alive_non_fallback_tokens = 0usize;
        let mut byte_fallback_tokens = 0usize;
        let mut pruned_tokens = 0usize;
        let mut atomic_char_alive = 0usize;
        let mut phase1_merged_alive = 0usize;
        let mut phase2_merged_alive = 0usize;
        let mut valid_utf8_learned_tokens = 0usize;
        let mut invalid_utf8_learned_tokens = 0usize;
        let mut breakdown = ScriptBreakdown::default();

        for tok in &self.internal_tokens {
            if tok.final_id.is_none() {
                pruned_tokens += 1;
                continue;
            }
            alive_vocab_size += 1;

            if tok.kind == TokenKind::ByteFallback {
                byte_fallback_tokens += 1;
                breakdown.byte_fallback += 1;
                continue;
            }

            alive_non_fallback_tokens += 1;
            let is_learned = matches!(
                tok.kind,
                TokenKind::AtomicChar | TokenKind::MergedPhase1 | TokenKind::MergedPhase2
            );
            match tok.kind {
                TokenKind::AtomicChar => atomic_char_alive += 1,
                TokenKind::MergedPhase1 => phase1_merged_alive += 1,
                TokenKind::MergedPhase2 => phase2_merged_alive += 1,
                TokenKind::Ascii | TokenKind::Special | TokenKind::ByteFallback => {}
            }

            match std::str::from_utf8(&tok.bytes) {
                Ok(text) => {
                    if is_learned {
                        valid_utf8_learned_tokens += 1;
                    }
                    classify_token_script(text, &mut breakdown);
                }
                Err(_) => {
                    invalid_utf8_learned_tokens += 1;
                    breakdown.invalid_utf8 += 1;
                }
            }
        }

        let invalid_utf8_rate = if alive_non_fallback_tokens == 0 {
            0.0
        } else {
            invalid_utf8_learned_tokens as f64 / alive_non_fallback_tokens as f64
        };

        Utf8AuditReport {
            total_internal_tokens: self.internal_tokens.len(),
            alive_vocab_size,
            alive_non_fallback_tokens,
            byte_fallback_tokens,
            pruned_tokens,
            atomic_char_alive,
            phase1_merged_alive,
            phase2_merged_alive,
            valid_utf8_learned_tokens,
            invalid_utf8_learned_tokens,
            invalid_utf8_rate,
            script_breakdown: breakdown,
        }
    }
}

fn is_hiragana(ch: char) -> bool {
    matches!(ch as u32, 0x3041..=0x3096 | 0x309D..=0x309F | 0x30FC)
}

fn is_katakana(ch: char) -> bool {
    matches!(ch as u32, 0x30A1..=0x30FA | 0x30FD..=0x30FF | 0xFF66..=0xFF9F | 0x30FC)
}

fn is_kanji(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F | 0x3005..=0x3007
    )
}

fn classify_token_script(text: &str, breakdown: &mut ScriptBreakdown) {
    if text.is_empty() {
        breakdown.mixed_utf8 += 1;
        return;
    }
    if text.is_ascii() {
        breakdown.ascii += 1;
        return;
    }

    let mut has_hira = false;
    let mut has_kata = false;
    let mut has_kanji = false;
    let mut has_other = false;

    for ch in text.chars() {
        if is_kanji(ch) {
            has_kanji = true;
        } else if ch == 'ー' {
            if has_kata {
                has_kata = true;
            } else if has_hira {
                has_hira = true;
            } else {
                has_other = true;
            }
        } else if is_hiragana(ch) {
            has_hira = true;
        } else if is_katakana(ch) {
            has_kata = true;
        } else {
            has_other = true;
        }
    }

    match (has_kanji, has_hira, has_kata, has_other) {
        (true, false, false, false) => breakdown.kanji += 1,
        (false, true, false, false) => breakdown.hiragana += 1,
        (false, false, true, false) => breakdown.katakana += 1,
        (true, true, false, false) => breakdown.kanji_hiragana += 1,
        _ => breakdown.mixed_utf8 += 1,
    }
}
