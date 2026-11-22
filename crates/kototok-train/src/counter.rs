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

/// Counts pre-token chunks, non-ASCII chars, and (optionally) Phase-2
/// segments across `text` in parallel.
pub fn count_corpus(text: &str, cfg: &TrainConfig) -> CorpusCounts {
    let blocks = line_blocks(text, rayon::current_num_threads().max(1) * 8);

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
                *chunks.entry(Box::from(b)).or_default() += 1;
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

    let mut counts = CorpusCounts {
        chunk_counts,
        char_counts,
        k2h_counts,
        ..CorpusCounts::default()
    };

    if cfg.phase2_enabled && !counts.chunk_counts.is_empty() {
        let mut table: Vec<Box<[u8]>> = counts.chunk_counts.keys().cloned().collect();
        table.sort();
        let id_of: FxHashMap<&[u8], u32> = table
            .iter()
            .enumerate()
            .map(|(i, b)| (b.as_ref(), i as u32))
            .collect();
        let max_seg = cfg.pretok.max_segment_chunks.max(1);

        let segments = blocks
            .par_iter()
            .map(|block| {
                let mut pts = Vec::new();
                pretokenize_into(block, &cfg.pretok, &mut pts);
                let mut segs: FxHashMap<Vec<u32>, u64> = FxHashMap::default();
                let mut cur: Vec<u32> = Vec::new();
                for pt in &pts {
                    cur.push(id_of[&block.as_bytes()[pt.start..pt.end]]);
                    if pt.hard_break_after || cur.len() >= max_seg {
                        if cur.len() >= 2 {
                            *segs.entry(std::mem::take(&mut cur)).or_default() += 1;
                        } else {
                            cur.clear();
                        }
                    }
                }
                if cur.len() >= 2 {
                    *segs.entry(cur).or_default() += 1;
                }
                segs
            })
            .reduce(FxHashMap::default, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_default() += v;
                }
                a
            });

        counts.chunk_table = table;
        counts.segments = segments;
    }

    counts
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
        let counts = count_corpus(&text, &c);
        assert!(counts.chunk_counts.values().sum::<u64>() >= 500);
        assert_eq!(counts.char_counts[&'新'], 100);
        assert!(!counts.segments.is_empty());
        assert!(counts.segments.keys().all(|s| s.len() >= 2));
        for seg in counts.segments.keys() {
            for &id in seg {
                assert!((id as usize) < counts.chunk_table.len());
            }
        }
    }

    #[test]
    fn empty_corpus_yields_empty_counts() {
        let counts = count_corpus("", &cfg());
        assert!(counts.chunk_counts.is_empty() && counts.segments.is_empty());
    }
}
