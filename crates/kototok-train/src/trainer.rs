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

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use kototok_core::{
    ByteMode, InternalToken, KTOK_FORMAT_VERSION, KototokError, Model, ModelMetadata, TokenKind,
    Tokenizer, VocabEvent, is_kanji_to_hiragana_cross,
};

use crate::config::TrainConfig;
use crate::counter::{CorpusCounts, count_corpus};
use crate::vocab_init::init_tokens;

const NIL: usize = usize::MAX;

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
    let counts = count_corpus(corpus, cfg);
    let (tokens, char_to_id) = init_tokens(cfg, &counts.char_counts)?;
    let mut trainer = Trainer::new(cfg.clone(), tokens, char_to_id);
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
    trainer.run(&counts)?;
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

/// Doubly-linked node in a word's symbol chain.
#[derive(Clone, Copy)]
struct Node {
    sym: u32,
    prev: usize,
    next: usize,
}

struct Word {
    head: usize,
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

    // Per-phase engine state (rebuilt by `load_words`).
    words: Vec<Word>,
    nodes: Vec<Node>,
    pair_counts: FxHashMap<(u32, u32), u64>,
    pair_to_words: FxHashMap<(u32, u32), FxHashSet<usize>>,
    sym_to_words: Vec<FxHashSet<usize>>,
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
            words: Vec::new(),
            nodes: Vec::new(),
            pair_counts: FxHashMap::default(),
            pair_to_words: FxHashMap::default(),
            sym_to_words: vec![FxHashSet::default(); n],
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

    fn run(&mut self, counts: &CorpusCounts) -> Result<(), TrainError> {
        self.run_phase1(counts);
        if self.cfg.phase2_enabled && self.alive_vocab_count < self.cfg.vocab_size {
            self.run_phase2(counts)?;
        }
        Ok(())
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

    fn run_phase1(&mut self, counts: &CorpusCounts) {
        let mut items = Vec::with_capacity(counts.chunk_counts.len());
        for (bytes, &weight) in &counts.chunk_counts {
            let syms = self.initial_ids(bytes);
            if !syms.is_empty() {
                items.push((syms, weight));
            }
        }
        self.load_words(items, Phase::One);
        self.run_merges(self.phase1_target(), Phase::One);
        self.phase1_vocab_size = self.alive_vocab_count;
        self.phase2_start_rank = self.events.len() as u32;
    }

    fn run_phase2(&mut self, counts: &CorpusCounts) -> Result<(), TrainError> {
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
        };
        let tok = Tokenizer::from_model(model)?;
        let enc: Vec<Vec<u32>> = counts
            .chunk_table
            .par_iter()
            .map(|b| tok.encode_phase1_internal(b).map(|v| v.to_vec()))
            .collect::<Result<_, _>>()?;

        let mut items = Vec::with_capacity(counts.segments.len());
        for (seq, &weight) in &counts.segments {
            let mut ids = Vec::new();
            for &cid in seq {
                ids.extend_from_slice(&enc[cid as usize]);
            }
            if ids.len() >= 2 {
                items.push((ids, weight));
            }
        }
        self.load_words(items, Phase::Two);
        self.run_merges(self.cfg.vocab_size, Phase::Two);
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

    /// Rebuilds the word/node/count engine for a phase.
    fn load_words(&mut self, items: Vec<(Vec<u32>, u64)>, phase: Phase) {
        self.words.clear();
        self.nodes.clear();
        self.pair_counts.clear();
        self.pair_to_words.clear();
        self.total_pairs = 0;
        self.heap.clear();
        self.sym_counts = vec![0; self.tokens.len()];
        self.sym_to_words = vec![FxHashSet::default(); self.tokens.len()];

        for (syms, weight) in items {
            if syms.is_empty() {
                continue;
            }
            let w = self.words.len();
            self.words.push(Word {
                head: self.nodes.len(),
                weight,
            });
            for (i, &s) in syms.iter().enumerate() {
                let next = if i + 1 == syms.len() {
                    NIL
                } else {
                    self.nodes.len() + 1
                };
                self.nodes.push(Node {
                    sym: s,
                    prev: if i == 0 { NIL } else { self.nodes.len() - 1 },
                    next,
                });
                self.sym_counts[s as usize] += weight;
                self.sym_to_words[s as usize].insert(w);
            }
            for win in syms.windows(2) {
                let p = (win[0], win[1]);
                *self.pair_counts.entry(p).or_default() += weight;
                self.pair_to_words.entry(p).or_default().insert(w);
                self.total_pairs += weight;
            }
        }

        let keys: Vec<(u32, u32)> = self.pair_counts.keys().copied().collect();
        for (l, r) in keys {
            self.push_candidate(l, r, phase);
        }
    }

    fn ensure_sym(&mut self, id: u32) {
        let i = id as usize;
        if self.sym_counts.len() <= i {
            self.sym_counts.resize(i + 1, 0);
            self.sym_to_words.resize_with(i + 1, FxHashSet::default);
        }
    }

    /// Collects the live symbol chain of a word.
    fn word_syms(&self, w: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut n = self.words[w].head;
        while n != NIL {
            out.push(self.nodes[n].sym);
            n = self.nodes[n].next;
        }
        out
    }

    fn sym_multiset(v: &[u32]) -> FxHashMap<u32, u64> {
        let mut m = FxHashMap::default();
        for &s in v {
            *m.entry(s).or_default() += 1;
        }
        m
    }

    fn pair_multiset(v: &[u32]) -> FxHashMap<(u32, u32), u64> {
        let mut m = FxHashMap::default();
        for win in v.windows(2) {
            *m.entry((win[0], win[1])).or_default() += 1;
        }
        m
    }

    /// Applies old->new word diffs to all count/membership indexes and pushes
    /// every newly present pair as a merge candidate.
    fn apply_diff(&mut self, w: usize, old_v: &[u32], new_v: &[u32], phase: Phase) {
        let weight = self.words[w].weight;
        let old_syms = Self::sym_multiset(old_v);
        let new_syms = Self::sym_multiset(new_v);
        let old_pairs = Self::pair_multiset(old_v);
        let new_pairs = Self::pair_multiset(new_v);

        for (&s, &oc) in &old_syms {
            let nc = new_syms.get(&s).copied().unwrap_or(0);
            if oc != nc {
                self.sym_counts[s as usize] -= oc * weight;
                self.sym_counts[s as usize] += nc * weight;
            }
            if nc == 0 {
                self.sym_to_words[s as usize].remove(&w);
            }
        }
        for (&s, &nc) in &new_syms {
            if !old_syms.contains_key(&s) {
                self.sym_counts[s as usize] += nc * weight;
                self.sym_to_words[s as usize].insert(w);
            }
        }

        for (&p, &oc) in &old_pairs {
            let nc = new_pairs.get(&p).copied().unwrap_or(0);
            if oc != nc {
                let e = self.pair_counts.entry(p).or_default();
                *e -= oc * weight;
                *e += nc * weight;
                self.total_pairs -= oc * weight;
                self.total_pairs += nc * weight;
            }
            if nc == 0
                && let Some(set) = self.pair_to_words.get_mut(&p)
            {
                set.remove(&w);
            }
        }
        for (&p, &nc) in &new_pairs {
            if !old_pairs.contains_key(&p) {
                *self.pair_counts.entry(p).or_default() += nc * weight;
                self.total_pairs += nc * weight;
                self.pair_to_words.entry(p).or_default().insert(w);
            }
        }
        for &(l, r) in new_pairs.keys() {
            self.push_candidate(l, r, phase);
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
    /// decides the class.
    fn phase2_npmi_floor(&self, l: u32, r: u32) -> f64 {
        if self.is_ascii_pair(l, r) {
            self.cfg.phase2_min_npmi_ascii
        } else {
            self.cfg.phase2_min_npmi
        }
    }

    fn is_ascii_pair(&self, l: u32, r: u32) -> bool {
        self.tokens[l as usize].bytes.iter().all(|b| b.is_ascii())
            && self.tokens[r as usize].bytes.iter().all(|b| b.is_ascii())
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

    /// BEG gate = `max(left-noun-gate, right-particle-gate)`: `0.0` lets the
    /// pair merge freely (stem + okurigana), `1.0` applies the full penalty
    /// (noun + particle). Unknown kanji suffixes default to noun.
    fn beg_gate(&self, l_bytes: &[u8], r_bytes: &[u8]) -> f64 {
        let suffix = crate::beg::trailing_kanji_run(l_bytes);
        let g_l = self.beg.suffix.get(suffix).copied().unwrap_or(1.0);
        let hira = crate::beg::leading_hiragana_run(r_bytes);
        let g_r = self.beg.hira.get(hira).copied().unwrap_or(0.0);
        g_l.max(g_r)
    }

    fn score(&self, l: u32, r: u32, f_ab: u64, phase: Phase) -> f64 {
        match phase {
            Phase::One => {
                let lb = &self.tokens[l as usize].bytes;
                let rb = &self.tokens[r as usize].bytes;
                if !is_kanji_to_hiragana_cross(lb, rb) {
                    return f_ab as f64;
                }
                let exp = if self.cfg.beg_strength > 0.0 {
                    self.cfg.beg_strength * self.beg_gate(lb, rb)
                } else if self.cfg.acsg_npmi_beta > 0.0 {
                    self.cfg.acsg_npmi_beta
                } else {
                    return f_ab as f64;
                };
                if exp <= 0.0 {
                    f_ab as f64
                } else {
                    let npmi = self.npmi(f_ab, l, r).clamp(0.01, 1.0);
                    f_ab as f64 * npmi.powf(exp)
                }
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
                }
                score
            }
        }
    }

    /// Lazy max-heap loop: pops the best-scored unseen pair, revalidates it,
    /// merges, then applies PickyBPE IoS pruning (Phase 1 only).
    fn run_merges(&mut self, target: usize, phase: Phase) {
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
        self.ensure_sym(new_id);
        self.seen.insert((l, r));
        self.events.push(VocabEvent::Merge {
            left: l,
            right: r,
            result: new_id,
        });
        let ws: Vec<usize> = self
            .pair_to_words
            .get(&(l, r))
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();
        for w in ws {
            self.merge_in_word(w, l, r, new_id, phase);
        }
        new_id
    }

    fn merge_in_word(&mut self, w: usize, l: u32, r: u32, new: u32, phase: Phase) {
        let old_v = self.word_syms(w);
        let mut n = self.words[w].head;
        while n != NIL {
            let m = self.nodes[n].next;
            if m != NIL && self.nodes[n].sym == l && self.nodes[m].sym == r {
                let p = self.nodes[n].prev;
                let nx = self.nodes[m].next;
                let k = self.nodes.len();
                self.nodes.push(Node {
                    sym: new,
                    prev: p,
                    next: nx,
                });
                if p != NIL {
                    self.nodes[p].next = k;
                } else {
                    self.words[w].head = k;
                }
                if nx != NIL {
                    self.nodes[nx].prev = k;
                }
                n = nx;
            } else {
                n = m;
            }
        }
        let new_v = self.word_syms(w);
        self.apply_diff(w, &old_v, &new_v, phase);
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
        let ws: Vec<usize> = self.sym_to_words[target as usize].iter().copied().collect();
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

    fn split_in_word(&mut self, w: usize, target: u32, split: &[u32], phase: Phase) {
        let old_v = self.word_syms(w);
        let mut n = self.words[w].head;
        while n != NIL {
            if self.nodes[n].sym != target {
                n = self.nodes[n].next;
                continue;
            }
            let p = self.nodes[n].prev;
            let nx = self.nodes[n].next;
            let mut prev_link = p;
            let mut first = NIL;
            let mut last = NIL;
            for &s in split {
                let k = self.nodes.len();
                self.nodes.push(Node {
                    sym: s,
                    prev: prev_link,
                    next: NIL,
                });
                if prev_link != NIL {
                    self.nodes[prev_link].next = k;
                } else {
                    self.words[w].head = k;
                }
                if first == NIL {
                    first = k;
                }
                prev_link = k;
                last = k;
            }
            if first == NIL {
                // Defensive: `split` is never empty in practice.
                if p != NIL {
                    self.nodes[p].next = nx;
                } else {
                    self.words[w].head = nx;
                }
                if nx != NIL {
                    self.nodes[nx].prev = p;
                }
            } else {
                self.nodes[last].next = nx;
                if nx != NIL {
                    self.nodes[nx].prev = last;
                }
            }
            n = nx;
        }
        let new_v = self.word_syms(w);
        self.apply_diff(w, &old_v, &new_v, phase);
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
        };
        let audit = model.audit_utf8_validity();
        model.metadata.vocab_size = audit.alive_vocab_size;
        model.metadata.pruned_count = audit.pruned_tokens;
        model.metadata.invalid_utf8_learned_tokens = audit.invalid_utf8_learned_tokens;
        model
    }
}
