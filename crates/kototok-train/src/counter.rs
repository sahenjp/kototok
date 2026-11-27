//! Parallel corpus statistics: pre-token chunks, atomic char frequencies,
//! and Phase-2 segment multiplicities.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use kototok_core::{ScriptClass, classify_char, pretokenize_into};

use crate::config::TrainConfig;

/// `(kanji_prefix, hiragana_run)` pair -> occurrence count.
pub type K2hCounts = FxHashMap<(Box<[u8]>, Box<[u8]>), u64>;

/// Aggregated corpus statistics used by vocabulary init and both phases.
#[derive(Default)]
pub struct CorpusCounts {
    /// Frequency of each unique pre-token byte slice.
    pub chunk_counts: FxHashMap<Box<[u8]>, u64>,
    /// Frequency of each non-ASCII `char` (drives `AtomicChar` selection).
    pub char_counts: FxHashMap<char, u64>,
    /// Deterministically ordered deduplicated chunk table (`Vec<u32>` ids in
    /// `segments` index into this). Populated only when Phase 2 is enabled.
    pub chunk_table: Vec<Box<[u8]>>,
    /// Segment multiplicity: `Vec` of chunk-table ids between hard breaks (or
    /// `max_segment_chunks`), keeping only segments with `>= 2` chunks.
    pub segments: FxHashMap<Vec<u32>, u64>,
    /// `(kanji_prefix, hiragana_run)` frequency inside `Kanji -> Hiragana`
    /// chunks; feeds the BEG branching-entropy gate. Populated only when
    /// `beg_strength > 0`.
    pub k2h_counts: K2hCounts,
}

/// Extracts `(kanji_prefix, first_hiragana_run)` at the first `Kanji ->
/// Hiragana` boundary inside a chunk, if any.
fn k2h_pair(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let s = std::str::from_utf8(bytes).ok()?;
    let mut kanji_end: Option<usize> = None;
    for (i, ch) in s.char_indices() {
        match classify_char(ch) {
            ScriptClass::Kanji => kanji_end = Some(i + ch.len_utf8()),
            ScriptClass::Hiragana => {
                if kanji_end == Some(i) {
                    let mut hend = i + ch.len_utf8();
                    for (_, ch2) in s[hend..].char_indices() {
                        if classify_char(ch2) == ScriptClass::Hiragana {
                            hend += ch2.len_utf8();
                        } else {
                            break;
                        }
                    }
                    return Some((&s.as_bytes()[..i], &s.as_bytes()[i..hend]));
                }
                kanji_end = None;
            }
            _ => kanji_end = None,
        }
    }
    None
}

/// Splits `text` into line-aligned, char-boundary-safe blocks of roughly
/// equal byte size for parallel processing.
fn line_blocks(text: &str, max_blocks: usize) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let target = (text.len() / max_blocks.max(1)).max(1);
    let mut blocks = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let mut want = (start + target).min(text.len());
        while want < text.len() && !text.is_char_boundary(want) {
            want += 1;
        }
        let end = match text[want..].find('\n') {
            Some(off) => want + off + 1,
            None => text.len(),
        };
        blocks.push(&text[start..end]);
        start = end;
    }
    blocks
}

/// Uses the same line-aligned boundaries for both counting passes.
pub(crate) fn corpus_blocks(text: &str) -> Vec<&str> {
    line_blocks(text, rayon::current_num_threads().max(1) * 8)
}

pub(crate) fn count_phase1(blocks: &[&str], cfg: &TrainConfig) -> CorpusCounts {
    let (chunk_counts, char_counts, k2h_counts) = blocks
        .par_iter()
        .map(|block| {
            let mut chunks: FxHashMap<Box<[u8]>, u64> = FxHashMap::default();
            let mut chars: FxHashMap<char, u64> = FxHashMap::default();
            let mut k2h: K2hCounts = FxHashMap::default();
            let beg = cfg.beg_strength > 0.0;
            let mut pts = Vec::new();
            pretokenize_into(block, &cfg.pretok, &mut pts);
            for pt in &pts {
                let b = &block.as_bytes()[pt.start..pt.end];
                if let Some(count) = chunks.get_mut(b) {
                    *count += 1;
                } else {
                    chunks.insert(Box::from(b), 1);
                }
                if beg && let Some((pre, hira)) = k2h_pair(b) {
                    *k2h.entry((Box::from(pre), Box::from(hira))).or_default() += 1;
                }
            }
            for ch in block.chars() {
                if !ch.is_ascii() {
                    *chars.entry(ch).or_default() += 1;
                }
            }
            (chunks, chars, k2h)
        })
        .reduce(
            || {
                (
                    FxHashMap::default(),
                    FxHashMap::default(),
                    FxHashMap::default(),
                )
            },
            |mut a, b| {
                for (k, v) in b.0 {
                    *a.0.entry(k).or_default() += v;
                }
                for (k, v) in b.1 {
                    *a.1.entry(k).or_default() += v;
                }
                for (k, v) in b.2 {
                    *a.2.entry(k).or_default() += v;
                }
                a
            },
        );

    CorpusCounts {
        chunk_counts,
        char_counts,
        k2h_counts,
        ..CorpusCounts::default()
    }
}

pub(crate) fn count_segments(
    blocks: &[&str],
    cfg: &TrainConfig,
    table: &[Box<[u8]>],
) -> FxHashMap<Box<[u32]>, u64> {
    let id_of: FxHashMap<&[u8], u32> = table
        .iter()
        .enumerate()
        .map(|(i, b)| (b.as_ref(), i as u32))
        .collect();
    // Preserve the training window cap independently of the runtime cap.
    let max_seg = cfg
        .pretok
        .max_segment_chunks
        .min(cfg.phase2_max_segment_chunks)
        .clamp(1, 16);
    let mut segments = blocks
        .par_iter()
        .map(|block| {
            let mut segments: FxHashMap<Box<[u32]>, u64> = FxHashMap::default();
            let mut pts = Vec::new();
            pretokenize_into(block, &cfg.pretok, &mut pts);
            let mut cur = Vec::with_capacity(max_seg);
            for pt in &pts {
                cur.push(id_of[&block.as_bytes()[pt.start..pt.end]]);
                if pt.hard_break_after || cur.len() >= max_seg {
                    if cur.len() >= 2 {
                        *segments.entry(Box::from(cur.as_slice())).or_default() += 1;
                    }
                    cur.clear();
                }
            }
            if cur.len() >= 2 {
                *segments.entry(Box::from(cur.as_slice())).or_default() += 1;
            }
            segments
        })
        // The reduction order sets word IDs for contextual candidate updates.
        .reduce(FxHashMap::default, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
            a
        });
    if cfg.min_segment_freq > 1 {
        if cfg.phase2_fallback_window >= 2 && cfg.phase2_fallback_window < max_seg {
            let sub_len = cfg.phase2_fallback_window;
            let mut kept = FxHashMap::default();
            let mut fallback: FxHashMap<Box<[u32]>, u64> = FxHashMap::default();
            for (seg, count) in segments {
                if count >= cfg.min_segment_freq {
                    kept.insert(seg, count);
                } else {
                    for chunk in seg.chunks(sub_len) {
                        if chunk.len() >= 2 {
                            *fallback.entry(Box::from(chunk)).or_default() += count;
                        }
                    }
                }
            }
            for (sub, count) in fallback {
                if count >= cfg.min_segment_freq {
                    *kept.entry(sub).or_default() += count;
                }
            }
            segments = kept;
        } else {
            segments.retain(|_, count| *count >= cfg.min_segment_freq);
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use kototok_core::PreTokenizerConfig;

    fn cfg() -> TrainConfig {
        TrainConfig {
            pretok: PreTokenizerConfig::default(),
            ..TrainConfig::default()
        }
    }

    #[test]
    fn counts_cover_corpus() {
        let c = cfg();
        let text = "新しいRustのトークナイザー。\n".repeat(100);
        let blocks = corpus_blocks(&text);
        let counts = count_phase1(&blocks, &c);
        let table: Vec<_> = counts.chunk_counts.keys().cloned().collect();
        let segments = count_segments(&blocks, &c, &table);
        assert!(counts.chunk_counts.values().sum::<u64>() >= 500);
        assert_eq!(counts.char_counts[&'新'], 100);
        assert!(!segments.is_empty());
        assert!(segments.keys().all(|s| s.len() >= 2));
        for seg in segments.keys() {
            for &id in seg {
                assert!((id as usize) < table.len());
            }
        }
    }

    #[test]
    fn empty_corpus_yields_empty_counts() {
        let blocks = corpus_blocks("");
        let counts = count_phase1(&blocks, &cfg());
        assert!(counts.chunk_counts.is_empty() && counts.segments.is_empty());
        assert!(count_segments(&blocks, &cfg(), &[]).is_empty());
    }

    #[test]
    fn segment_frequency_is_filtered_after_combining_blocks() {
        let cfg = crate::TrainPreset::Global.config(512);
        let blocks = ["abc def\n", "abc ghi\n", "abc def\n"];
        let counts = count_phase1(&blocks, &cfg);
        let table: Vec<_> = counts.chunk_counts.into_keys().collect();
        let segments = count_segments(&blocks, &cfg, &table);
        assert_eq!(segments.len(), 1);
        let (ids, &weight) = segments.iter().next().unwrap();
        let chunks: Vec<_> = ids.iter().map(|&id| table[id as usize].as_ref()).collect();
        assert_eq!(chunks, [b"abc".as_slice(), b" def".as_slice(), b"\n"]);
        assert_eq!(weight, 2);
    }
}
