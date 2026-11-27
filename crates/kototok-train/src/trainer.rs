//! Incremental vocabulary trainer.
//!
//! Phase 1 runs lazy-heap BPE over unique pre-token chunks with ACSG NPMI
//! scoring for `Kanji -> Hiragana` cross pairs and PickyBPE IoS pruning of
//! absorbed `MergedPhase1` intermediates. Phase 2 runs controlled superword
//! merges over unique hard-bounded segments of Phase-1-encoded ids.
//! Every mutation is recorded as a chronological `VocabEvent` so the core
//! `Tokenizer` can replay it exactly.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::collections::hash_map::Entry;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use kototok_core::{
    ByteMode, InternalToken, KTOK_FORMAT_VERSION, KototokError, Model, ModelMetadata, ScriptClass,
    TokenKind, Tokenizer, VocabEvent, classify_char, is_japanese_bytes, is_kanji_to_hiragana_cross,
};

use crate::config::TrainConfig;
use crate::counter::{corpus_blocks, count_phase1, count_segments};
use crate::vocab_init::init_tokens;

/// Errors produced by [`train`].
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    /// `vocab_size` cannot hold the 256 byte entries plus special tokens.
    #[error("vocab_size {0} cannot hold the base vocabulary (256 bytes + specials)")]
    VocabTooSmall(usize),
    /// The intermediate Phase-1 model failed core validation.
    #[error(transparent)]
    Core(#[from] KototokError),
}

/// Trains a `Model` on `corpus` according to `cfg`.
pub fn train(corpus: &str, cfg: &TrainConfig) -> Result<Model, TrainError> {
    let started = std::env::var_os("KOTOTOK_PROGRESS").map(|_| Instant::now());
    let blocks = corpus_blocks(corpus);
    let mut counts = count_phase1(&blocks, cfg);
    let (tokens, char_to_id) = init_tokens(cfg, &counts.char_counts)?;
    let mut trainer = Trainer::new(cfg.clone(), tokens, char_to_id);
    trainer.started = started;
    if cfg.beg_strength > 0.0 {
        trainer.beg = crate::beg::build_beg_scores(
            &counts,
            cfg.beg_stem_max,
            cfg.beg_noun_min,
            cfg.beg_particle_lo,
            cfg.beg_particle_hi,
        );
        if std::env::var_os("KOTOTOK_BEG_DEBUG").is_some() {
            let mut v: Vec<_> = trainer.beg.suffix.iter().collect();
            v.sort_by(|a, b| a.1.total_cmp(b.1));
            eprintln!("[beg] {} suffix keys", v.len());
            for (k, g) in v.iter().take(15) {
                eprintln!("[beg-lo ] {g:>5.3}  {:?}", String::from_utf8_lossy(k));
            }
            for (k, g) in v.iter().rev().take(15) {
                eprintln!("[beg-hi ] {g:>5.3}  {:?}", String::from_utf8_lossy(k));
            }
            let mut h: Vec<_> = trainer.beg.hira.iter().collect();
            h.sort_by(|a, b| a.1.total_cmp(b.1));
            eprintln!("[beg] {} hira keys", h.len());
            for (k, g) in h.iter().rev().take(20) {
                eprintln!("[beg-hira] {g:>5.3}  {:?}", String::from_utf8_lossy(k));
            }
        }
    }
    let chunks = std::mem::take(&mut counts.chunk_counts);
    drop(counts);
    trainer.run_phase1(&chunks);
    if cfg.phase2_enabled && trainer.alive_vocab_count < cfg.vocab_size {
        trainer.drop_phase_state();
        let mut table: Vec<_> = chunks.into_keys().collect();
        table.sort_unstable();
        trainer.progress("phase2 counting");
        let segments = count_segments(&blocks, cfg, &table);
        if started.is_some() {
            eprintln!("[train] phase2 retained_segments={}", segments.len());
        }
        trainer.run_phase2(table, segments)?;
    }
    Ok(trainer.finalize())
}

/// Total-order `f64` wrapper for the deterministic max-heap.
#[derive(Clone, Copy, PartialEq)]
struct Score(f64);

impl Eq for Score {}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

type Heap = BinaryHeap<(Score, Reverse<(u32, u32)>)>;
type Pair = (u32, u32);

const POSTING_INSERT: u32 = 0;
const POSTING_REMOVE: u32 = 1;

struct Word {
    syms: Vec<u32>,
    weight: u64,
}

/// Which training phase produced a merge / drives scoring rules.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    One,
    Two,
}

struct Trainer {
    cfg: TrainConfig,
    tokens: Vec<InternalToken>,
    alive: Vec<bool>,
    parents: Vec<Option<(u32, u32)>>,
    alive_vocab_count: usize,
    /// Pairs already consumed by a `VocabEvent::Merge`; never re-emitted.
    seen: FxHashSet<(u32, u32)>,
    events: Vec<VocabEvent>,
    char_to_id: FxHashMap<char, u32>,
    phase1_vocab_size: usize,
    phase2_start_rank: u32,
    started: Option<Instant>,

    // Per-phase engine state (rebuilt by `load_words`).
    words: Vec<Word>,
    pair_counts: FxHashMap<Pair, u64>,
    pair_to_words: FxHashMap<Pair, Vec<u32>>,
    // Contextual PickyBPE may restore a vanished pair. Preserve its
    // posting history outside the live index so its traversal is unchanged.
    empty_pair_postings: FxHashMap<Pair, Vec<u32>>,
    contextual_picky: bool,
    // Only Phase-1 merges from a PickyBPE run get a posting list.
    first_prunable_id: usize,
    sym_to_words: Vec<Vec<u32>>,
    sym_counts: Vec<u64>,
    total_pairs: u64,
    heap: Heap,
    /// BEG: two-sided gate tables for `Kanji -> Hiragana` merge scoring.
    beg: crate::beg::BegTables,
    /// Phase-2 vocabulary budget (set at `run_phase2` entry) and the count
    /// of pure-ASCII merges already spent from it.
    phase2_budget: usize,
    phase2_ascii_merges: usize,
}

impl Trainer {
    fn new(cfg: TrainConfig, tokens: Vec<InternalToken>, char_to_id: FxHashMap<char, u32>) -> Self {
        let n = tokens.len();
        Self {
            cfg,
            tokens,
            alive: vec![true; n],
            parents: vec![None; n],
            alive_vocab_count: n,
            seen: FxHashSet::default(),
            events: Vec::new(),
            char_to_id,
            phase1_vocab_size: n,
            phase2_start_rank: 0,
            started: None,
            words: Vec::new(),
            pair_counts: FxHashMap::default(),
            pair_to_words: FxHashMap::default(),
            empty_pair_postings: FxHashMap::default(),
            contextual_picky: false,
            first_prunable_id: n,
            sym_to_words: Vec::new(),
            sym_counts: vec![0; n],
            phase2_budget: 0,
            phase2_ascii_merges: 0,
            beg: crate::beg::BegTables {
                suffix: FxHashMap::default(),
                hira: FxHashMap::default(),
            },
            total_pairs: 0,
            heap: BinaryHeap::new(),
        }
    }

    fn progress(&self, status: &str) {
        if let Some(started) = self.started {
            eprintln!(
                "[train] {status} vocab_count={} events={} pair_counts={} elapsed_secs={:.2}",
                self.alive_vocab_count,
                self.events.len(),
                self.pair_counts.len(),
                started.elapsed().as_secs_f64(),
            );
        }
    }

    fn phase1_target(&self) -> usize {
        if self.cfg.phase2_enabled {
            ((self.cfg.vocab_size as f64) * self.cfg.phase1_vocab_ratio)
                .round()
                .clamp(0.0, self.cfg.vocab_size as f64) as usize
        } else {
            self.cfg.vocab_size
        }
    }

    fn run_phase1(&mut self, chunks: &FxHashMap<Box<[u8]>, u64>) {
        let words = chunks
            .iter()
            .map(|(bytes, &weight)| Word {
                syms: self.initial_ids(bytes),
                weight,
            })
            .collect();
        self.load_words(words, Phase::One);
        self.progress("phase1 start");
        self.run_merges(self.phase1_target(), Phase::One);
        self.progress("phase1 end");
        self.phase1_vocab_size = self.alive_vocab_count;
        self.phase2_start_rank = self.events.len() as u32;
    }

    fn drop_phase_state(&mut self) {
        self.words = Vec::new();
        self.pair_counts = FxHashMap::default();
        self.pair_to_words = FxHashMap::default();
        self.empty_pair_postings = FxHashMap::default();
        self.sym_to_words = Vec::new();
        self.sym_counts = Vec::new();
        self.total_pairs = 0;
        self.heap = Heap::new();
    }

    fn run_phase2(
        &mut self,
        table: Vec<Box<[u8]>>,
        segments: FxHashMap<Box<[u32]>, u64>,
    ) -> Result<(), TrainError> {
        self.phase2_budget = self.cfg.vocab_size.saturating_sub(self.alive_vocab_count);
        self.phase2_ascii_merges = 0;
        // Compile a temporary Phase-1-only tokenizer to encode every unique
        // chunk to internal ids (parallel over the deduplicated table).
        let mut toks = self.tokens.clone();
        for (i, t) in toks.iter_mut().enumerate() {
            t.final_id = if self.alive[i] { Some(0) } else { None };
        }
        let model = Model {
            metadata: ModelMetadata {
                name: self.cfg.name.clone(),
                version: KTOK_FORMAT_VERSION,
                vocab_size: toks.len(),
                phase1_vocab_size: self.phase1_vocab_size,
                phase2_start_rank: self.events.len() as u32,
                pruned_count: 0,
                invalid_utf8_learned_tokens: 0,
                train_notes: String::new(),
            },
            pretok: self.cfg.pretok.clone(),
            special_tokens: self.special_pairs(),
            internal_tokens: toks,
            events: self.events.clone(),
            tag_ranges: Vec::new(),
            tagged_prunes: Vec::new(),
        };
        let tok = Tokenizer::from_model(model)?;
        let mut used = vec![false; table.len()];
        for seq in segments.keys() {
            for &cid in seq.iter() {
                used[cid as usize] = true;
            }
        }
        let enc: Vec<Vec<u32>> = table
            .into_par_iter()
            .zip(used.into_par_iter())
            .map(|(bytes, used)| {
                if used {
                    tok.encode_phase1_internal(&bytes).map(|v| v.to_vec())
                } else {
                    Ok(Vec::new())
                }
            })
            .collect::<Result<_, _>>()?;
        drop(tok);

        let mut words = Vec::with_capacity(segments.len());
        for (seq, weight) in segments {
            let len = seq.iter().map(|&cid| enc[cid as usize].len()).sum();
            if len >= 2 {
                let mut syms = Vec::with_capacity(len);
                for &cid in seq.iter() {
                    syms.extend_from_slice(&enc[cid as usize]);
                }
                words.push(Word { syms, weight });
            }
        }
        drop(enc);
        self.load_words(words, Phase::Two);
        self.progress("phase2 start");
        self.run_merges(self.cfg.vocab_size, Phase::Two);
        self.progress("phase2 end");
        Ok(())
    }

    fn special_pairs(&self) -> Vec<(String, u32)> {
        self.cfg
            .special_tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), 256 + i as u32))
            .collect()
    }

    /// Maps a chunk's raw bytes to initial internal ids (mirrors
    /// `Tokenizer::phase1_uncached` pre-replay mapping).
    fn initial_ids(&self, bytes: &[u8]) -> Vec<u32> {
        match self.cfg.pretok.byte_mode {
            ByteMode::RawBytes => bytes.iter().map(|&b| b as u32).collect(),
            ByteMode::Utf8Atomic => match std::str::from_utf8(bytes) {
                Ok(s) => {
                    let mut out = Vec::with_capacity(bytes.len());
                    for ch in s.chars() {
                        if (ch as u32) < 128 {
                            out.push(ch as u32);
                        } else if let Some(&id) = self.char_to_id.get(&ch) {
                            out.push(id);
                        } else {
                            let mut buf = [0u8; 4];
                            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                                out.push(b as u32);
                            }
                        }
                    }
                    out
                }
                Err(_) => bytes.iter().map(|&b| b as u32).collect(),
            },
        }
    }

    /// `Special` never merges; `ByteFallback` merges only in `RawBytes` mode.
    fn is_mergeable(&self, id: u32) -> bool {
        match self.tokens[id as usize].kind {
            TokenKind::Special => false,
            TokenKind::ByteFallback => self.cfg.pretok.byte_mode == ByteMode::RawBytes,
            _ => true,
        }
    }

    /// Builds weighted counts and postings for a fresh phase.
    fn load_words(&mut self, words: Vec<Word>, phase: Phase) {
        let contextual = self.has_contextual_scores(phase);
        self.contextual_picky = contextual && phase == Phase::One && self.cfg.picky_enabled;
        self.sym_counts = vec![0; self.tokens.len()];
        if phase == Phase::Two {
            // Pass 1: exact symbol counts, total_pairs, and initial pair counts.
            // Phase 2 never prunes, so pre-existing pairs below `min_pair_freq`
            // can never become mergeable and need no postings.
            for word in &words {
                for &sym in &word.syms {
                    self.sym_counts[sym as usize] += word.weight;
                }
                for pair in word.syms.windows(2) {
                    self.add_pair((pair[0], pair[1]), word.weight);
                }
            }
            let min_f = self.cfg.min_pair_freq;
            self.pair_counts.retain(|_, count| *count >= min_f);
            self.pair_counts.shrink_to_fit();

            let mut active = Vec::new();
            for word in words {
                if !word
                    .syms
                    .windows(2)
                    .any(|p| self.pair_counts.contains_key(&(p[0], p[1])))
                {
                    continue;
                }
                let w = active.len() as u32;
                for pair in word.syms.windows(2) {
                    let pair = (pair[0], pair[1]);
                    if self.pair_counts.contains_key(&pair) {
                        self.record_pair(pair, w, contextual);
                    }
                }
                active.push(word);
            }
            self.words = active;
        } else {
            for (w, word) in words.iter().enumerate() {
                for &sym in &word.syms {
                    self.sym_counts[sym as usize] += word.weight;
                }
                for pair in word.syms.windows(2) {
                    let pair = (pair[0], pair[1]);
                    self.add_pair(pair, word.weight);
                    self.record_pair(pair, w as u32, contextual);
                }
            }
            self.words = words;
        }

        let keys: Vec<_> = self.pair_counts.keys().copied().collect();
        for (l, r) in keys {
            self.push_candidate(l, r, phase);
        }
    }

    fn add_pair(&mut self, pair: Pair, weight: u64) -> bool {
        let is_new = match self.pair_counts.entry(pair) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() += weight;
                false
            }
            Entry::Vacant(entry) => {
                entry.insert(weight);
                true
            }
        };
        self.total_pairs += weight;
        is_new
    }

    fn record_pair(&mut self, pair: Pair, word: u32, initial: bool) {
        if self.contextual_picky {
            if self.seen.contains(&pair) {
                return;
            }
            let words = self
                .pair_to_words
                .entry(pair)
                .or_insert_with(|| self.empty_pair_postings.remove(&pair).unwrap_or_default());
            if initial || !words.ends_with(&[POSTING_INSERT, word]) {
                words.extend([POSTING_INSERT, word]);
            }
        } else {
            let words = self.pair_to_words.entry(pair).or_default();
            // The legacy index inserted each initial occurrence, including
            // duplicates, which can grow a full hash set before its lookup.
            if initial || words.last() != Some(&word) {
                words.push(word);
            }
        }
    }

    fn remove_pair(&mut self, pair: Pair, weight: u64) {
        if let Some(count) = self.pair_counts.get_mut(&pair) {
            *count = count.saturating_sub(weight);
            if *count == 0 {
                self.pair_counts.remove(&pair);
                if !self.contextual_picky {
                    self.pair_to_words.remove(&pair);
                }
            }
        }
        self.total_pairs -= weight;
    }

    fn remove_pair_posting(&mut self, pair: Pair, word: u32) {
        if let Some(words) = self.pair_to_words.get_mut(&pair) {
            words.extend([POSTING_REMOVE, word]);
            if !self.pair_counts.contains_key(&pair) {
                let history = self.pair_to_words.remove(&pair).unwrap();
                self.empty_pair_postings.insert(pair, history);
            }
        }
    }

    fn record_prunable(&mut self, sym: u32, word: u32) {
        if let Some(words) = (sym as usize)
            .checked_sub(self.first_prunable_id)
            .and_then(|index| self.sym_to_words.get_mut(index))
        {
            if self.contextual_picky {
                if !words.ends_with(&[POSTING_INSERT, word]) {
                    words.extend([POSTING_INSERT, word]);
                }
            } else if words.last() != Some(&word) {
                words.push(word);
            }
        }
    }

    fn has_contextual_scores(&self, phase: Phase) -> bool {
        phase == Phase::Two
            || self.cfg.beg_strength > 0.0
            || self.cfg.acsg_npmi_beta > 0.0
            || self.cfg.beg_okuri_boost > 0.0
    }

    /// Contextual scores are refreshed after each affected word, so their
    /// lazy heap depends on the original hash-set traversal order.
    fn order_postings(&self, ws: &mut Vec<u32>, contextual: bool) {
        if ws.len() < 2 {
            return;
        }
        if contextual {
            let mut words = FxHashSet::default();
            if self.contextual_picky {
                for &[op, word] in ws.as_chunks::<2>().0 {
                    if op == POSTING_INSERT {
                        words.insert(word);
                    } else {
                        words.remove(&word);
                    }
                }
            } else {
                for &w in ws.iter() {
                    words.insert(w);
                }
            }
            ws.clear();
            ws.extend(words);
        } else {
            if !ws.is_sorted() {
                ws.sort_unstable();
            }
            ws.dedup();
        }
    }

    fn push_candidate(&mut self, l: u32, r: u32, phase: Phase) {
        if self.seen.contains(&(l, r)) || !self.is_mergeable(l) || !self.is_mergeable(r) {
            return;
        }
        let f = self.pair_counts.get(&(l, r)).copied().unwrap_or(0);
        if f < self.cfg.min_pair_freq {
            return;
        }
        if self.beg_hard_veto(l, r) {
            return;
        }
        if phase == Phase::Two {
            if self.npmi(f, l, r) < self.phase2_npmi_floor(l, r) {
                return;
            }
            if self.is_ascii_pair(l, r) && self.phase2_ascii_quota_reached() {
                return;
            }
        }
        let s = self.score(l, r, f, phase);
        self.heap.push((Score(s), Reverse((l, r))));
    }

    /// Class-scoped Phase-2 NPMI floor: pairs made entirely of ASCII bytes
    /// (pure-Latin superwords) may merge at `phase2_min_npmi_ascii`, buying
    /// English compression without touching any JA-script boundary. UTF-8
    /// guarantees JA-script bytes are all >= 0x80, so an ASCII byte scan
    /// decides the class. `Kanji -> Hiragana` pairs whose right side reads as
    /// an inflectional run (gate_R < 0.9) get their own low floor — whole-word
    /// conjugated forms — while particle-like runs are already vetoed and the
    /// mid band keeps the strict floor.
    fn phase2_npmi_floor(&self, l: u32, r: u32) -> f64 {
        if self.is_ascii_pair(l, r) {
            return self.cfg.phase2_min_npmi_ascii;
        }
        let lb = &self.tokens[l as usize].bytes;
        let rb = &self.tokens[r as usize].bytes;
        if self.cfg.phase2_non_ja_loose && !is_japanese_bytes(lb) && !is_japanese_bytes(rb) {
            return self.cfg.phase2_min_npmi_ascii;
        }
        if is_kanji_to_hiragana_cross(lb, rb) {
            let hira = crate::beg::leading_hiragana_run(rb);
            let gate_r = self.beg.hira.get(hira).copied().unwrap_or(0.0);
            if gate_r < 0.9 {
                return self.cfg.phase2_min_npmi_okuri;
            }
            return self.cfg.phase2_min_npmi;
        }
        // Okurigana-shaped pair with a partly-merged stem (`確か|める`):
        // right is all-Hiragana, left is a Kanji+Hiragana word. Gate on the
        // COMBINED hiragana tail — the run the corpus actually shows after
        // the kanji stem — so `名詞|の` stays on the strict floor.
        if let Some(gate) = self.okurigana_tail_gate(lb, rb)
            && gate < 0.9
        {
            return self.cfg.phase2_min_npmi_okuri;
        }
        // Ren'yo compound (`書き|換える`): left ends in Hiragana and holds a
        // Kanji stem; right is Kanji-led with a Hiragana tail (verb-shaped).
        if Self::is_renyo_pair(lb, rb) {
            return self.cfg.phase2_min_npmi_okuri;
        }
        self.cfg.phase2_min_npmi
    }

    /// Gate for pairs whose merge extends a stem+okurigana word (`確か|める`):
    /// `rb` all-Hiragana, `lb` a Kanji/Hiragana word; the looked-up tail is
    /// the combined Hiragana run. Returns `None` unless the run is attested.
    fn okurigana_tail_gate(&self, lb: &[u8], rb: &[u8]) -> Option<f64> {
        if !crate::beg::all_hiragana(rb) || !crate::beg::is_kanji_hiragana_word(lb) {
            return None;
        }
        let mut tail = crate::beg::trailing_hiragana_run(lb).to_vec();
        tail.extend_from_slice(rb);
        self.beg.hira.get(tail.as_slice()).copied()
    }

    /// `lb` ends in Hiragana and contains a Kanji; `rb` starts with Kanji and
    /// contains Hiragana — the shape of a ren'yo (verb-stem) compound.
    fn is_renyo_pair(lb: &[u8], rb: &[u8]) -> bool {
        let (Ok(ls), Ok(rs)) = (std::str::from_utf8(lb), std::str::from_utf8(rb)) else {
            return false;
        };
        let mut lb_kanji = false;
        let mut lb_ends_hira = false;
        for ch in ls.chars() {
            let cl = classify_char(ch);
            lb_kanji |= cl == ScriptClass::Kanji;
            lb_ends_hira = cl == ScriptClass::Hiragana;
        }
        let mut it = rs.chars();
        let Some(first) = it.next() else {
            return false;
        };
        let rb_verb_shaped = classify_char(first) == ScriptClass::Kanji
            && it.any(|c| classify_char(c) == ScriptClass::Hiragana);
        lb_kanji && lb_ends_hira && rb_verb_shaped
    }

    fn is_ascii_pair(&self, l: u32, r: u32) -> bool {
        self.tokens[l as usize].bytes.iter().all(|b| b.is_ascii())
            && self.tokens[r as usize].bytes.iter().all(|b| b.is_ascii())
    }

    /// Hard veto: a `Kanji -> Hiragana` pair whose right side is a definite
    /// particle-like run (gate_R ~= 1) may never merge. The multiplicative
    /// NPMI gate alone is too weak for very high-NPMI pairs — `npmi^3` still
    /// leaves ~73% of the score, so a large vocab budget lets 日本|の slip
    /// through. Inflectional runs (べる, しい) sit below 0.9 and stay legal.
    fn beg_hard_veto(&self, l: u32, r: u32) -> bool {
        if self.cfg.beg_strength <= 0.0
            || !is_kanji_to_hiragana_cross(
                &self.tokens[l as usize].bytes,
                &self.tokens[r as usize].bytes,
            )
        {
            return false;
        }
        let hira = crate::beg::leading_hiragana_run(&self.tokens[r as usize].bytes);
        self.beg.hira.get(hira).copied().unwrap_or(0.0) >= 0.95
    }

    /// Whether the ASCII share of the Phase-2 budget is exhausted. A quota
    /// keeps cheap Latin superwords from crowding out JA Phase-2 merges in
    /// a fixed-size vocabulary.
    fn phase2_ascii_quota_reached(&self) -> bool {
        self.cfg.phase2_ascii_quota < 1.0
            && self.phase2_ascii_merges
                >= (self.phase2_budget as f64 * self.cfg.phase2_ascii_quota) as usize
    }

    /// NPMI of pair `(l, r)` against current symbol counts and `total_pairs`.
    /// Returns `1.0` when the pair is deterministic (`f_ab >= T`).
    fn npmi(&self, f_ab: u64, l: u32, r: u32) -> f64 {
        let f_l = self.sym_counts[l as usize];
        let f_r = self.sym_counts[r as usize];
        let t = self.total_pairs;
        if f_ab == 0 || f_l == 0 || f_r == 0 || t == 0 {
            return 0.0;
        }
        if f_ab >= t {
            return 1.0;
        }
        let pmi = ((f_ab as f64 * t as f64) / (f_l as f64 * f_r as f64)).ln();
        pmi / -(f_ab as f64 / t as f64).ln()
    }

    /// BEG gate = `max(left-noun-gate, right-particle-gate)` unless the
    /// right side is clearly inflectional (`g_r <= 0.5`), in which case the
    /// right gate alone decides: a productive stem like `染` keeps ~10
    /// distinct inflectional followers and would otherwise read as
    /// noun-like. Unknown kanji suffixes default to noun.
    fn score(&self, l: u32, r: u32, f_ab: u64, phase: Phase) -> f64 {
        match phase {
            Phase::One => {
                if !self.has_contextual_scores(phase) {
                    return f_ab as f64;
                }
                let lb = &self.tokens[l as usize].bytes;
                let rb = &self.tokens[r as usize].bytes;
                if !is_kanji_to_hiragana_cross(lb, rb) {
                    // Partly-merged stems (`確か|める`) are okurigana-shaped
                    // but never cross K->H at the boundary, so the boost must
                    // fire here too for stem+inflection forms to win slots.
                    if self.cfg.beg_okuri_boost > 0.0
                        && let Some(gate) = self.okurigana_tail_gate(lb, rb)
                        && gate < 0.9
                    {
                        return f_ab as f64 * (1.0 + self.cfg.beg_okuri_boost * (1.0 - gate));
                    }
                    return f_ab as f64;
                }
                // Right side is the reliable signal (same lesson as Phase
                // 2): a low-gate Hiragana run is inflectional, so the pair is
                // stem+okurigana no matter how diverse the stem's follower
                // set is. `max(g_l, g_r)` punishes productive stems like `染`
                // because verbs take ~10+ inflection shapes -> high entropy.
                // Only when the right side is NOT clearly inflectional does
                // the left-side noun gate matter.
                let suffix = crate::beg::trailing_kanji_run(lb);
                let g_l = self.beg.suffix.get(suffix).copied().unwrap_or(1.0);
                let hira = crate::beg::leading_hiragana_run(rb);
                let g_r = self.beg.hira.get(hira).copied().unwrap_or(0.0);
                let gate = if self.cfg.beg_strength > 0.0 && g_r <= 0.5 {
                    g_r
                } else {
                    g_l.max(g_r)
                };
                let exp = if self.cfg.beg_strength > 0.0 {
                    self.cfg.beg_strength * gate
                } else if self.cfg.acsg_npmi_beta > 0.0 {
                    self.cfg.acsg_npmi_beta
                } else {
                    return f_ab as f64;
                };
                let base = if exp <= 0.0 {
                    f_ab as f64
                } else {
                    let npmi = self.npmi(f_ab, l, r).clamp(0.01, 1.0);
                    f_ab as f64 * npmi.powf(exp)
                };
                // Boost is keyed on the right run only: an inflectional
                // right side promotes the merge even when the left stem has
                // a high-entropy follower distribution.
                base * (1.0 + self.cfg.beg_okuri_boost * (1.0 - g_r))
            }
            Phase::Two => {
                let npmi = self.npmi(f_ab, l, r);
                let mut score = f_ab as f64 * (1.0 + npmi);
                // BEG on soft-boundary superwords uses only the RIGHT-side
                // gate (is the right operand a particle-like Hiragana run?).
                // Phase-1 may split stem+okurigana (e.g. (食)|(べる)) on a
                // shared inflection; Phase-2 superwords are what rejoin them,
                // so an inflectional right side must stay cheap to merge.
                // Noun+particle pairs (日本|の) are caught because particles
                // carry gate_R ~1 regardless of the left operand.
                if self.cfg.beg_strength > 0.0
                    && self.cfg.beg_phase2
                    && is_kanji_to_hiragana_cross(
                        &self.tokens[l as usize].bytes,
                        &self.tokens[r as usize].bytes,
                    )
                {
                    let hira = crate::beg::leading_hiragana_run(&self.tokens[r as usize].bytes);
                    let g_r = self.beg.hira.get(hira).copied().unwrap_or(0.0);
                    let exp = self.cfg.beg_strength * g_r;
                    if exp > 0.0 {
                        score *= npmi.clamp(0.01, 1.0).powf(exp);
                    }
                    // Same-side boost: inflectional right runs (低 gate_R)
                    // get promoted so stem+inflection superwords re-form.
                    score *= 1.0 + self.cfg.beg_okuri_boost * (1.0 - g_r);
                } else if self.cfg.beg_okuri_boost > 0.0
                    && let Some(gate) = self.okurigana_tail_gate(
                        &self.tokens[l as usize].bytes,
                        &self.tokens[r as usize].bytes,
                    )
                    && gate < 0.9
                {
                    score *= 1.0 + self.cfg.beg_okuri_boost * (1.0 - gate);
                } else if Self::is_renyo_pair(
                    &self.tokens[l as usize].bytes,
                    &self.tokens[r as usize].bytes,
                ) {
                    // Ren'yo compounds (`書き|換える`) get the full boost:
                    // no inflectional run exists to gate on, and these pairs
                    // are rare enough that a half-strength bonus never wins
                    // the heap ordering against frequent superwords. The
                    // verb-shaped right side is itself the license.
                    score *= 1.0 + self.cfg.beg_okuri_boost;
                }
                score
            }
        }
    }

    /// Lazy max-heap loop: pops the best-scored unseen pair, revalidates it,
    /// merges, then applies PickyBPE IoS pruning (Phase 1 only).
    fn run_merges(&mut self, target: usize, phase: Phase) {
        let mut merges = 0usize;
        while self.alive_vocab_count < target {
            let Some((Score(s), Reverse((l, r)))) = self.heap.pop() else {
                break;
            };
            if self.seen.contains(&(l, r))
                || !self.alive[l as usize]
                || !self.alive[r as usize]
                || !self.is_mergeable(l)
                || !self.is_mergeable(r)
            {
                continue;
            }
            let f_ab = self.pair_counts.get(&(l, r)).copied().unwrap_or(0);
            if f_ab < self.cfg.min_pair_freq {
                continue;
            }
            let exact = self.score(l, r, f_ab, phase);
            if exact < s {
                // Stale optimistic entry: reinsert at its real score.
                self.heap.push((Score(exact), Reverse((l, r))));
                continue;
            }
            if phase == Phase::Two {
                if self.npmi(f_ab, l, r) < self.phase2_npmi_floor(l, r) {
                    continue;
                }
                if self.is_ascii_pair(l, r) && self.phase2_ascii_quota_reached() {
                    continue;
                }
            }

            // IoS uses pre-merge symbol frequencies.
            let ios_l = f_ab as f64 / self.sym_counts[l as usize].max(1) as f64;
            let ios_r = f_ab as f64 / self.sym_counts[r as usize].max(1) as f64;
            if phase == Phase::Two && self.is_ascii_pair(l, r) {
                self.phase2_ascii_merges += 1;
            }
            self.do_merge(l, r, phase);

            if phase == Phase::One && self.cfg.picky_enabled {
                if self.tokens[l as usize].kind == TokenKind::MergedPhase1
                    && self.alive[l as usize]
                    && ios_l >= self.cfg.picky_ios_threshold
                {
                    self.do_prune(l, phase);
                }
                if r != l
                    && self.tokens[r as usize].kind == TokenKind::MergedPhase1
                    && self.alive[r as usize]
                    && ios_r >= self.cfg.picky_ios_threshold
                {
                    self.do_prune(r, phase);
                }
            }
            merges += 1;
            if merges.is_multiple_of(25_000) {
                self.progress(match phase {
                    Phase::One => "phase1 merge",
                    Phase::Two => "phase2 merge",
                });
            }
        }
    }

    /// Executes a merge: records the event, appends the token, splices all
    /// non-overlapping `(l, r)` sites inside every affected word.
    fn do_merge(&mut self, l: u32, r: u32, phase: Phase) -> u32 {
        let new_id = self.tokens.len() as u32;
        let mut bytes = self.tokens[l as usize].bytes.clone();
        bytes.extend_from_slice(&self.tokens[r as usize].bytes);
        self.tokens.push(InternalToken {
            internal_id: new_id,
            bytes,
            kind: if phase == Phase::One {
                TokenKind::MergedPhase1
            } else {
                TokenKind::MergedPhase2
            },
            final_id: None,
        });
        self.alive.push(true);
        self.parents.push(Some((l, r)));
        self.alive_vocab_count += 1;
        self.sym_counts.push(0);
        if phase == Phase::One && self.cfg.picky_enabled {
            self.sym_to_words.push(Vec::new());
        }
        self.seen.insert((l, r));
        self.events.push(VocabEvent::Merge {
            left: l,
            right: r,
            result: new_id,
        });
        let mut ws = self.pair_to_words.remove(&(l, r)).unwrap_or_default();
        let contextual = self.has_contextual_scores(phase);
        self.order_postings(&mut ws, contextual);
        let mut new_pairs = Vec::new();
        let mut removed_pairs = Vec::new();
        for w in ws {
            let mut syms = std::mem::take(&mut self.words[w as usize].syms);
            if self.merge_in_word(
                w,
                &mut syms,
                (l, r),
                new_id,
                &mut new_pairs,
                &mut removed_pairs,
            ) && contextual
            {
                // NPMI depends on all symbol counts, including for pairs
                // whose frequency did not change in this word.
                for pair in syms.windows(2) {
                    self.push_candidate(pair[0], pair[1], phase);
                }
            }
            self.words[w as usize].syms = syms;
        }
        if !contextual {
            // These pairs contain a fresh id so their counts only increased.
            for (left, right) in new_pairs {
                self.push_candidate(left, right, phase);
            }
        }
        new_id
    }

    /// Compacts left-to-right, charging the original edges between groups
    /// once, even when neighboring groups both collapse.
    fn merge_in_word(
        &mut self,
        word: u32,
        syms: &mut Vec<u32>,
        pair: (u32, u32),
        new: u32,
        new_pairs: &mut Vec<(u32, u32)>,
        removed_pairs: &mut Vec<Pair>,
    ) -> bool {
        let weight = self.words[word as usize].weight;
        let (l, r) = pair;
        let mut read = 0;
        let mut write = 0;
        let mut previous_input = 0;
        let mut previous_merged = false;
        let mut merged = 0;
        while read < syms.len() {
            let first = syms[read];
            let collapse = first == l && syms.get(read + 1) == Some(&r);
            let output = if collapse {
                self.remove_pair(pair, weight);
                merged += 1;
                new
            } else {
                first
            };
            if write > 0 && (previous_merged || collapse) {
                let removed = (previous_input, first);
                self.remove_pair(removed, weight);
                if self.contextual_picky {
                    removed_pairs.push(removed);
                }
                let formed = (syms[write - 1], output);
                if self.add_pair(formed, weight) {
                    new_pairs.push(formed);
                }
                self.record_pair(formed, word, false);
            }
            syms[write] = output;
            previous_merged = collapse;
            previous_input = if collapse { r } else { first };
            read += 1 + usize::from(collapse);
            write += 1;
        }
        if merged == 0 {
            return false;
        }
        syms.truncate(write);
        let delta = merged * weight;
        self.sym_counts[l as usize] -= delta;
        self.sym_counts[r as usize] -= delta;
        self.sym_counts[new as usize] += delta;
        if self.contextual_picky {
            for removed in removed_pairs.drain(..) {
                if !syms.windows(2).any(|pair| (pair[0], pair[1]) == removed) {
                    self.remove_pair_posting(removed, word);
                }
            }
            for sym in [l, r] {
                if let Some(index) = (sym as usize).checked_sub(self.first_prunable_id)
                    && !syms.contains(&sym)
                {
                    self.sym_to_words[index].extend([POSTING_REMOVE, word]);
                }
            }
        }
        self.record_prunable(new, word);
        true
    }

    /// Executes a PickyBPE prune: expands `target`'s parents to alive tokens,
    /// records the event, frees the vocab slot, and splices the expansion
    /// into every word that still contains `target`.
    fn do_prune(&mut self, target: u32, phase: Phase) {
        let Some((pl, pr)) = self.parents[target as usize] else {
            return;
        };
        let mut split = Vec::new();
        self.expand_alive(pl, &mut split);
        self.expand_alive(pr, &mut split);
        self.events.push(VocabEvent::Prune {
            target,
            split_into: split.clone(),
        });
        self.alive[target as usize] = false;
        self.alive_vocab_count -= 1;
        let mut ws =
            std::mem::take(&mut self.sym_to_words[target as usize - self.first_prunable_id]);
        self.order_postings(&mut ws, self.contextual_picky);
        for w in ws {
            self.split_in_word(w, target, &split, phase);
        }
    }

    /// Expands a token to the currently-alive leaves of its merge tree.
    fn expand_alive(&self, id: u32, out: &mut Vec<u32>) {
        if self.alive[id as usize] || self.parents[id as usize].is_none() {
            out.push(id);
        } else {
            let (l, r) = self.parents[id as usize].unwrap();
            self.expand_alive(l, out);
            self.expand_alive(r, out);
        }
    }

    fn split_in_word(&mut self, word: u32, target: u32, split: &[u32], phase: Phase) {
        let w = word as usize;
        let weight = self.words[w].weight;
        let mut syms = std::mem::take(&mut self.words[w].syms);
        let original_len = syms.len();
        let mut occurrences = 0;
        let mut old_pairs = self.contextual_picky.then(|| {
            syms.windows(2)
                .map(|pair| (pair[0], pair[1]))
                .collect::<FxHashSet<_>>()
        });
        for pair in syms.windows(2) {
            if pair[0] == target || pair[1] == target {
                let removed = (pair[0], pair[1]);
                self.remove_pair(removed, weight);
                if self.contextual_picky {
                    self.remove_pair_posting(removed, word);
                }
            }
        }
        for &sym in &syms {
            occurrences += usize::from(sym == target);
        }
        if occurrences != 0 {
            let delta = occurrences as u64 * weight;
            self.sym_counts[target as usize] -= delta;
            for &sym in split {
                self.sym_counts[sym as usize] += delta;
                if !self.contextual_picky || !syms.contains(&sym) {
                    self.record_prunable(sym, word);
                }
            }

            let new_len = original_len + occurrences * (split.len() - 1);
            syms.resize(new_len, 0);
            let mut write = new_len;
            let mut right_expanded = false;
            for read in (0..original_len).rev() {
                let sym = syms[read];
                let expanded = sym == target;
                let replacement = if expanded {
                    split
                } else {
                    std::slice::from_ref(&sym)
                };
                let boundary = (write < new_len && (expanded || right_expanded))
                    .then(|| (replacement[replacement.len() - 1], syms[write]));
                for pair in replacement
                    .windows(2)
                    .map(|pair| (pair[0], pair[1]))
                    .chain(boundary)
                {
                    self.add_pair(pair, weight);
                    if old_pairs.as_mut().is_none_or(|pairs| pairs.insert(pair)) {
                        self.record_pair(pair, word, false);
                    }
                }
                let start = write - replacement.len();
                syms[start..write].copy_from_slice(replacement);
                write = start;
                right_expanded = expanded;
            }
            for pair in syms.windows(2) {
                self.push_candidate(pair[0], pair[1], phase);
            }
        }
        self.words[w].syms = syms;
    }

    /// Assigns dense `final_id`s to alive tokens in `internal_id` order and
    /// assembles the audited `Model`.
    fn finalize(mut self) -> Model {
        let mut fid = 0u32;
        for (i, t) in self.tokens.iter_mut().enumerate() {
            t.final_id = if self.alive[i] {
                let f = fid;
                fid += 1;
                Some(f)
            } else {
                None
            };
        }
        let mut model = Model {
            metadata: ModelMetadata {
                name: self.cfg.name.clone(),
                version: KTOK_FORMAT_VERSION,
                vocab_size: 0,
                phase1_vocab_size: self.phase1_vocab_size,
                phase2_start_rank: self.phase2_start_rank,
                pruned_count: 0,
                invalid_utf8_learned_tokens: 0,
                train_notes: format!("kototok-train config '{}'", self.cfg.name),
            },
            pretok: self.cfg.pretok.clone(),
            special_tokens: self.special_pairs(),
            internal_tokens: self.tokens,
            events: self.events,
            tag_ranges: Vec::new(),
            tagged_prunes: Vec::new(),
        };
        let audit = model.audit_utf8_validity();
        model.metadata.vocab_size = audit.alive_vocab_size;
        model.metadata.pruned_count = audit.pruned_tokens;
        model.metadata.invalid_utf8_learned_tokens = audit.invalid_utf8_learned_tokens;
        model
    }
}
