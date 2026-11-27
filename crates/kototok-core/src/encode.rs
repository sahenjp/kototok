//! Event-ordered tokenizer encoder.
//!
//! Replay semantics are strictly chronological (PickyBPE): merge and prune
//! events are executed in rank order off a min-heap over a doubly-linked node
//! arena, and executing an event at rank `r` may only enqueue events with
//! `rank > r`. This makes merge/prune cycles mathematically impossible and
//! gives worst-case `O(N)` behavior per Phase-1 chunk plus per Phase-2
//! segment.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::decode as dec;
use crate::model::{ByteMode, KototokError, Model, TokenKind, VocabEvent};
use crate::pretok::pretokenize_into;

const NIL: usize = usize::MAX;
const KIND_MERGE: u8 = 0;
const KIND_PRUNE: u8 = 1;
const PHASE1_CACHE_CAP: usize = 32_768;
const PHASE1_CACHE_MAX_CHUNK: usize = 128;

static NEXT_UID: AtomicU64 = AtomicU64::new(1);

/// Phase-1 result cache: tokenizer uid -> (chunk bytes -> internal ids).
type Phase1Cache = FxHashMap<u64, FxHashMap<Box<[u8]>, SmallVec<[u32; 8]>>>;

thread_local! {
    /// Per-thread Phase-1 chunk cache, namespaced by `Tokenizer::uid`.
    static PHASE1_CACHE: RefCell<Phase1Cache> = RefCell::new(FxHashMap::default());
}

type Heap = BinaryHeap<Reverse<(u32, usize, u8, usize)>>;

#[derive(Clone, Copy)]
struct Node {
    id: u32,
    order: usize,
    prev: usize,
    next: usize,
    alive: bool,
}

/// A compiled `kototok` tokenizer. Cheap to share: it is `Send + Sync` and
/// keeps no mutable state, so `&Tokenizer` is safe across Rayon threads.
pub struct Tokenizer {
    model: Model,
    char_to_id: FxHashMap<char, u32>,
    /// (left, right) -> (rank, result) for `VocabEvent::Merge`.
    pair_to_merge: FxHashMap<(u32, u32), (u32, u32)>,
    /// internal_id -> (rank, split_into) for `VocabEvent::Prune`.
    prune_table: Vec<Option<(u32, Vec<u32>)>>,
    internal_to_final: Vec<Option<u32>>,
    final_to_bytes: Vec<Vec<u8>>,
    phase2_start_rank: u32,
    has_phase2: bool,
    has_prunes: bool,
    uid: u64,
}

impl Tokenizer {
    /// Compiles a `Model` into an executable `Tokenizer`.
    pub fn from_model(model: Model) -> Result<Self, KototokError> {
        model.validate_final_id_sparsity()?;
        let n_internal = model.internal_tokens.len();
        if n_internal < 256 {
            return Err(KototokError::InvalidFormat(
                "internal token table must contain at least 256 byte-level entries".into(),
            ));
        }
        for (idx, tok) in model.internal_tokens.iter().enumerate() {
            if tok.internal_id as usize != idx {
                return Err(KototokError::InvalidFormat(format!(
                    "internal token table index {idx} has mismatched internal_id {}",
                    tok.internal_id
                )));
            }
        }
        for (b, tok) in model.internal_tokens.iter().enumerate().take(256) {
            if tok.bytes != [b as u8] {
                return Err(KototokError::InvalidFormat(format!(
                    "byte-level entry {b} does not map to its raw byte"
                )));
            }
        }

        let mut char_to_id = FxHashMap::default();
        for tok in &model.internal_tokens {
            if tok.kind == TokenKind::AtomicChar {
                let s = std::str::from_utf8(&tok.bytes).map_err(|_| {
                    KototokError::InvalidFormat(format!(
                        "AtomicChar {} has invalid UTF-8 bytes",
                        tok.internal_id
                    ))
                })?;
                let mut chars = s.chars();
                match (chars.next(), chars.next()) {
                    (Some(ch), None) => {
                        char_to_id.insert(ch, tok.internal_id);
                    }
                    _ => {
                        return Err(KototokError::InvalidFormat(format!(
                            "AtomicChar {} is not exactly one character",
                            tok.internal_id
                        )));
                    }
                }
            }
        }

        let mut pair_to_merge = FxHashMap::default();
        let mut prune_table: Vec<Option<(u32, Vec<u32>)>> = vec![None; n_internal];
        let mut has_prunes = false;
        for (rank, ev) in model.events.iter().enumerate() {
            let rank = rank as u32;
            match ev {
                VocabEvent::Merge {
                    left,
                    right,
                    result,
                } => {
                    for id in [*left, *right, *result] {
                        if id as usize >= n_internal {
                            return Err(KototokError::InvalidFormat(format!(
                                "merge event {rank} references internal id {id} out of range"
                            )));
                        }
                    }
                    pair_to_merge.insert((*left, *right), (rank, *result));
                }
                VocabEvent::Prune { target, split_into } => {
                    if *target as usize >= n_internal
                        || split_into.iter().any(|id| *id as usize >= n_internal)
                    {
                        return Err(KototokError::InvalidFormat(format!(
                            "prune event {rank} references internal id out of range"
                        )));
                    }
                    has_prunes = true;
                    let slot = &mut prune_table[*target as usize];
                    match slot {
                        // Keep the earliest chronological prune for a target.
                        Some((r, _)) if *r <= rank => {}
                        _ => *slot = Some((rank, split_into.clone())),
                    }
                }
            }
        }

        let internal_to_final: Vec<Option<u32>> = model
            .internal_tokens
            .iter()
            .map(|tok| tok.final_id)
            .collect();
        let final_to_bytes: Vec<Vec<u8>> = model
            .final_id_to_bytes()
            .into_iter()
            .map(|b| b.to_vec())
            .collect();

        let phase2_start_rank = model.metadata.phase2_start_rank;
        let has_phase2 = phase2_start_rank < model.events.len() as u32;
        let uid = NEXT_UID.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            model,
            char_to_id,
            pair_to_merge,
            prune_table,
            internal_to_final,
            final_to_bytes,
            phase2_start_rank,
            has_phase2,
            has_prunes,
            uid,
        })
    }

    /// The underlying model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[inline]
    fn try_enqueue_merge(
        &self,
        nodes: &[Node],
        heap: &mut Heap,
        idx: usize,
        min_rank: u32,
        rank_limit: u32,
    ) {
        let n = &nodes[idx];
        let m = n.next;
        if m == NIL {
            return;
        }
        if let Some(&(rank, _result)) = self.pair_to_merge.get(&(n.id, nodes[m].id))
            && rank >= min_rank
            && rank < rank_limit
        {
            heap.push(Reverse((rank, n.order, KIND_MERGE, idx)));
        }
    }

    #[inline]
    fn try_enqueue_prune(
        &self,
        nodes: &[Node],
        heap: &mut Heap,
        idx: usize,
        min_rank: u32,
        rank_limit: u32,
    ) {
        if !self.has_prunes {
            return;
        }
        if let Some(Some((rank, _))) = self.prune_table.get(nodes[idx].id as usize) {
            let rank = *rank;
            if rank >= min_rank && rank < rank_limit {
                heap.push(Reverse((rank, nodes[idx].order, KIND_PRUNE, idx)));
            }
        }
    }

    /// Runs the monotonic event engine over `initial` internal ids.
    ///
    /// Only events with `rank_min <= rank < rank_limit` execute; events created
    /// while executing rank `r` must have `rank > r`, guaranteeing
    /// chronological termination.
    fn run_engine(&self, initial: &[u32], rank_min: u32, rank_limit: u32) -> SmallVec<[u32; 8]> {
        let mut nodes: Vec<Node> = Vec::with_capacity(initial.len().saturating_mul(2) + 8);
        for (pos, &id) in initial.iter().enumerate() {
            nodes.push(Node {
                id,
                order: pos,
                prev: if pos == 0 { NIL } else { pos - 1 },
                next: if pos + 1 == initial.len() {
                    NIL
                } else {
                    pos + 1
                },
                alive: true,
            });
        }
        let mut head = if initial.is_empty() { NIL } else { 0 };
        let mut heap: Heap = BinaryHeap::new();
        for idx in 0..initial.len() {
            self.try_enqueue_merge(&nodes, &mut heap, idx, rank_min, rank_limit);
            self.try_enqueue_prune(&nodes, &mut heap, idx, rank_min, rank_limit);
        }

        while let Some(Reverse((rank, _order, kind, idx))) = heap.pop() {
            match kind {
                KIND_MERGE => {
                    let n = nodes[idx];
                    if !n.alive || n.next == NIL {
                        continue;
                    }
                    let m = n.next;
                    if !nodes[m].alive {
                        continue;
                    }
                    let Some(&(r, result)) = self.pair_to_merge.get(&(n.id, nodes[m].id)) else {
                        continue;
                    };
                    if r != rank {
                        continue;
                    }
                    // Splice merged node in place of (idx, m).
                    let p = n.prev;
                    let nx = nodes[m].next;
                    let new_idx = nodes.len();
                    nodes.push(Node {
                        id: result,
                        order: n.order,
                        prev: p,
                        next: nx,
                        alive: true,
                    });
                    nodes[idx].alive = false;
                    nodes[m].alive = false;
                    if p != NIL {
                        nodes[p].next = new_idx;
                    } else {
                        head = new_idx;
                    }
                    if nx != NIL {
                        nodes[nx].prev = new_idx;
                    }
                    // Chronological enqueue: only rank > cur_rank.
                    let min = rank + 1;
                    self.try_enqueue_merge(&nodes, &mut heap, new_idx, min, rank_limit);
                    if p != NIL {
                        self.try_enqueue_merge(&nodes, &mut heap, p, min, rank_limit);
                    }
                    self.try_enqueue_prune(&nodes, &mut heap, new_idx, min, rank_limit);
                }
                _ => {
                    // KIND_PRUNE
                    let n = nodes[idx];
                    if !n.alive {
                        continue;
                    }
                    let Some(Some((r, split))) = self.prune_table.get(n.id as usize) else {
                        continue;
                    };
                    if *r != rank {
                        continue;
                    }
                    let split = split.clone();
                    let p = n.prev;
                    let nx = n.next;
                    nodes[idx].alive = false;

                    let mut prev_link = p;
                    let mut first_new = NIL;
                    let mut last_new = NIL;
                    for &sid in &split {
                        let ni = nodes.len();
                        nodes.push(Node {
                            id: sid,
                            order: n.order,
                            prev: prev_link,
                            next: NIL,
                            alive: true,
                        });
                        if prev_link != NIL {
                            nodes[prev_link].next = ni;
                        }
                        if first_new == NIL {
                            first_new = ni;
                        }
                        prev_link = ni;
                        last_new = ni;
                    }
                    if first_new == NIL {
                        // Empty split: node just disappears.
                        if p != NIL {
                            nodes[p].next = nx;
                        } else {
                            head = nx;
                        }
                        if nx != NIL {
                            nodes[nx].prev = p;
                        }
                    } else {
                        nodes[last_new].next = nx;
                        if nx != NIL {
                            nodes[nx].prev = last_new;
                        }
                        if p == NIL {
                            head = first_new;
                        }
                    }

                    let min = rank + 1;
                    if p != NIL {
                        self.try_enqueue_merge(&nodes, &mut heap, p, min, rank_limit);
                    }
                    // Enqueue pair events for each spliced node and any prunes
                    // that now apply to them.
                    let mut cur = first_new;
                    while cur != NIL {
                        self.try_enqueue_merge(&nodes, &mut heap, cur, min, rank_limit);
                        self.try_enqueue_prune(&nodes, &mut heap, cur, min, rank_limit);
                        cur = nodes[cur].next;
                        if cur == nx {
                            break;
                        }
                    }
                }
            }
        }

        let mut out = SmallVec::new();
        let mut cur = head;
        while cur != NIL {
            out.push(nodes[cur].id);
            cur = nodes[cur].next;
        }
        out
    }

    /// Phase-1 encode of a single pre-token chunk into internal ids.
    ///
    /// Results are memoized in a bounded thread-local cache keyed by this
    /// tokenizer's `uid`; chunks over 128 bytes bypass the cache.
    pub fn encode_phase1_internal(&self, chunk: &[u8]) -> Result<SmallVec<[u32; 8]>, KototokError> {
        if chunk.len() > PHASE1_CACHE_MAX_CHUNK {
            return Ok(self.phase1_uncached(chunk));
        }
        if let Some(hit) = PHASE1_CACHE.with(|c| {
            c.borrow()
                .get(&self.uid)
                .and_then(|inner| inner.get(chunk))
                .cloned()
        }) {
            return Ok(hit);
        }
        let result = self.phase1_uncached(chunk);
        PHASE1_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            let inner = cache.entry(self.uid).or_default();
            if inner.len() >= PHASE1_CACHE_CAP {
                inner.clear();
            }
            inner.insert(Box::from(chunk), result.clone());
        });
        Ok(result)
    }

    /// Initial internal ids for a chunk + Phase-1 event replay
    /// (`rank < phase2_start_rank`).
    fn phase1_uncached(&self, chunk: &[u8]) -> SmallVec<[u32; 8]> {
        let mut ids: SmallVec<[u32; 8]> = SmallVec::with_capacity(chunk.len());
        match self.model.pretok.byte_mode {
            ByteMode::Utf8Atomic => match std::str::from_utf8(chunk) {
                Ok(s) => {
                    for ch in s.chars() {
                        let c = ch as u32;
                        if c < 128 {
                            ids.push(c);
                        } else if let Some(&id) = self.char_to_id.get(&ch) {
                            ids.push(id);
                        } else {
                            let mut buf = [0u8; 4];
                            for &b in ch.encode_utf8(&mut buf).as_bytes() {
                                ids.push(b as u32);
                            }
                        }
                    }
                }
                // Not valid UTF-8: degrade cleanly to raw bytes.
                Err(_) => ids.extend(chunk.iter().map(|&b| b as u32)),
            },
            ByteMode::RawBytes => ids.extend(chunk.iter().map(|&b| b as u32)),
        }
        self.run_engine(&ids, 0, self.phase2_start_rank)
    }

    /// Phase-2 encode: concatenated Phase-1 ids within a soft-bounded segment.
    fn phase2_segment(&self, ids: &[u32]) -> SmallVec<[u32; 8]> {
        if !self.has_phase2 && !self.has_prunes {
            return SmallVec::from_slice(ids);
        }
        let limit = self.model.events.len() as u32;
        self.run_engine(ids, self.phase2_start_rank, limit)
    }

    #[inline]
    fn internal_to_final_id(&self, internal: u32) -> Result<u32, KototokError> {
        match self.internal_to_final.get(internal as usize) {
            Some(Some(fid)) => Ok(*fid),
            Some(None) => Err(KototokError::PrunedTokenEmitted(internal)),
            None => Err(KototokError::InvalidTokenId(internal)),
        }
    }

    /// Runs pretokenize + Phase 1 + Phase 2 and returns final token ids.
    fn encode_final_ids(&self, text: &str) -> Result<Vec<u32>, KototokError> {
        let mut pts = Vec::new();
        pretokenize_into(text, &self.model.pretok, &mut pts);
        let max_seg = self.model.pretok.max_segment_chunks.max(1);

        let mut out = Vec::new();
        let mut segment: Vec<u32> = Vec::new();
        let mut seg_chunks = 0usize;

        let flush_segment = |tok: &Tokenizer, segment: &mut Vec<u32>, out: &mut Vec<u32>| {
            if segment.is_empty() {
                return Ok(());
            }
            for id in tok.phase2_segment(segment).iter().copied() {
                out.push(tok.internal_to_final_id(id)?);
            }
            segment.clear();
            Ok::<(), KototokError>(())
        };

        for pt in &pts {
            let ids = self.encode_phase1_internal(&text.as_bytes()[pt.start..pt.end])?;
            segment.extend_from_slice(&ids);
            seg_chunks += 1;
            if pt.hard_break_after || seg_chunks >= max_seg {
                flush_segment(self, &mut segment, &mut out)?;
                seg_chunks = 0;
            }
        }
        flush_segment(self, &mut segment, &mut out)?;
        Ok(out)
    }

    /// Encodes text to final token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, KototokError> {
        self.encode_final_ids(text)
    }

    /// Encodes text to `(final_id, byte_start, byte_end)` spans computed from
    /// each token's byte length.
    pub fn encode_with_offsets(
        &self,
        text: &str,
    ) -> Result<Vec<(u32, usize, usize)>, KototokError> {
        let ids = self.encode_final_ids(text)?;
        let mut pos = 0usize;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let len = self
                .final_to_bytes
                .get(id as usize)
                .map(|b| b.len())
                .ok_or(KototokError::InvalidTokenId(id))?;
            out.push((id, pos, pos + len));
            pos += len;
        }
        Ok(out)
    }

    /// Encodes text and returns each token's bytes as lossy strings.
    pub fn tokenize_pieces(&self, text: &str) -> Result<Vec<String>, KototokError> {
        let ids = self.encode_final_ids(text)?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let bytes = self
                .final_to_bytes
                .get(id as usize)
                .ok_or(KototokError::InvalidTokenId(id))?;
            out.push(String::from_utf8_lossy(bytes).into_owned());
        }
        Ok(out)
    }

    /// Encodes a batch of texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, KototokError> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Decodes final ids to raw bytes.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, KototokError> {
        dec::decode_bytes(&self.final_to_bytes, ids)
    }

    /// Decodes final ids to a `String`.
    pub fn decode(&self, ids: &[u32]) -> Result<String, KototokError> {
        dec::decode(&self.final_to_bytes, ids)
    }

    /// Lossy decode that never fails; unknown ids are skipped.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        dec::decode_lossy(&self.final_to_bytes, ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{InternalToken, ModelMetadata, PreTokenizerConfig, TokenKind};
    use crate::script::is_kanji_to_hiragana_cross;

    fn byte_tokens() -> Vec<InternalToken> {
        (0..256u32)
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
            .collect()
    }

    fn base_model(mut extra: Vec<InternalToken>, events: Vec<VocabEvent>, phase2: u32) -> Model {
        let mut internal = byte_tokens();
        internal.append(&mut extra);
        let vocab = internal.iter().filter(|t| t.final_id.is_some()).count();
        Model {
            metadata: ModelMetadata {
                name: "test".into(),
                version: 1,
                vocab_size: vocab,
                phase1_vocab_size: 256,
                phase2_start_rank: phase2,
                pruned_count: internal.iter().filter(|t| t.final_id.is_none()).count(),
                invalid_utf8_learned_tokens: 0,
                train_notes: String::new(),
            },
            pretok: PreTokenizerConfig::default(),
            special_tokens: vec![],
            internal_tokens: internal,
            events,
            tag_ranges: Vec::new(),
            tagged_prunes: Vec::new(),
        }
    }

    fn merged(id: u32, bytes: &[u8], kind: TokenKind, alive: bool) -> InternalToken {
        InternalToken {
            internal_id: id,
            bytes: bytes.to_vec(),
            kind,
            final_id: if alive { Some(id) } else { None },
        }
    }

    fn atomic(id: u32, s: &str) -> InternalToken {
        InternalToken {
            internal_id: id,
            bytes: s.as_bytes().to_vec(),
            kind: TokenKind::AtomicChar,
            final_id: Some(id),
        }
    }

    #[test]
    fn tokenizer_rejects_sparse_final_id() {
        let mut model = base_model(vec![], vec![], 0);
        model.internal_tokens[0].final_id = Some(u32::MAX - 1);
        assert!(matches!(
            Tokenizer::from_model(model),
            Err(KototokError::InvalidFormat(_))
        ));
    }

    #[test]
    fn ascii_and_fallback_roundtrip() {
        let tok = Tokenizer::from_model(base_model(vec![], vec![], 0)).unwrap();
        let ids = tok.encode("hello あ").unwrap();
        // 'h','e','l','l','o',' ' as ascii ids; あ falls back to its UTF-8 bytes
        let mut expect: Vec<u32> = b"hello ".iter().map(|&b| b as u32).collect();
        expect.extend("あ".as_bytes().iter().map(|&b| b as u32));
        assert_eq!(ids, expect);
        assert_eq!(tok.decode(&ids).unwrap(), "hello あ");
    }

    #[test]
    fn atomic_char_used_when_known() {
        let model = base_model(vec![atomic(256, "あ")], vec![], 0);
        let tok = Tokenizer::from_model(model).unwrap();
        let ids = tok.encode("あ").unwrap();
        assert_eq!(ids, vec![256]);
        assert_eq!(tok.decode(&ids).unwrap(), "あ");
    }

    #[test]
    fn merge_prune_merge_replay_is_monotonic() {
        // Chronology: merge a+b -> AB (rank 0), prune AB -> [a,b] (rank 1),
        // merge b+c -> BC (rank 2).
        let model = base_model(
            vec![
                merged(256, b"ab", TokenKind::MergedPhase1, false), // pruned
                merged(257, b"bc", TokenKind::MergedPhase1, true),
            ],
            vec![
                VocabEvent::Merge {
                    left: 97,
                    right: 98,
                    result: 256,
                },
                VocabEvent::Prune {
                    target: 256,
                    split_into: vec![97, 98],
                },
                VocabEvent::Merge {
                    left: 98,
                    right: 99,
                    result: 257,
                },
            ],
            3, // all events are phase 1
        );
        let tok = Tokenizer::from_model(model).unwrap();
        // "ab": a,b -> AB(256) at rank 0; prune at rank 1 splices back -> [97,98]
        let ids = tok.encode("ab").unwrap();
        assert_eq!(ids, vec![97, 98]);
        // "abc": a,b->AB rank0; AB pruned rank1 -> a,b,c; b,c->BC rank2 -> a,BC
        let ids = tok.encode("abc").unwrap();
        assert_eq!(ids, vec![97, 257]);
        assert_eq!(tok.decode(&ids).unwrap(), "abc");
    }

    #[test]
    fn pruned_token_survival_errors() {
        // A merge produces a token with no final_id and no prune event.
        let model = base_model(
            vec![merged(256, b"ab", TokenKind::MergedPhase1, false)],
            vec![VocabEvent::Merge {
                left: 97,
                right: 98,
                result: 256,
            }],
            1,
        );
        let tok = Tokenizer::from_model(model).unwrap();
        let err = tok.encode("ab").unwrap_err();
        assert!(matches!(err, KototokError::PrunedTokenEmitted(256)));
    }

    #[test]
    fn phase2_merges_across_soft_not_hard() {
        // Phase 1 (ranks 0-1): "1"+"2" -> "12" (256), "12"+"3" -> "123" (257)
        // Phase 2 (rank 2):    "123"+"4" -> "1234" (258)
        let model = base_model(
            vec![
                merged(256, b"12", TokenKind::MergedPhase1, true),
                merged(257, b"123", TokenKind::MergedPhase1, true),
                merged(258, b"1234", TokenKind::MergedPhase2, true),
            ],
            vec![
                VocabEvent::Merge {
                    left: 49,
                    right: 50,
                    result: 256,
                },
                VocabEvent::Merge {
                    left: 256,
                    right: 51,
                    result: 257,
                },
                VocabEvent::Merge {
                    left: 257,
                    right: 52,
                    result: 258,
                },
            ],
            2,
        );
        let tok = Tokenizer::from_model(model).unwrap();
        // "1234" pretokenizes as "123"|"4" (soft): phase2 merges across.
        assert_eq!(tok.encode("1234").unwrap(), vec![258]);
        // "123.4": '.' hard-splits, phase2 can never merge across it.
        let dot = b'.' as u32;
        assert_eq!(tok.encode("123.4").unwrap(), vec![257, dot, 52]);
        // "123\n4": newline hard-splits.
        assert_eq!(tok.encode("123\n4").unwrap(), vec![257, 10, 52]);
    }

    #[test]
    fn japanese_phase2_segment_boundaries() {
        // Phase1: h(104)+i(105) -> "hi" (256)
        // Phase2: "hi" + '!'(33) -> "hi!" (257)  -- '!' is AsciiPunctHard
        // Also test crossing 。 never allowed: use merge "hi"+x.
        let model = base_model(
            vec![
                merged(256, b"hi", TokenKind::MergedPhase1, true),
                merged(257, b"hi!", TokenKind::MergedPhase2, true),
                merged(258, b"hix", TokenKind::MergedPhase2, true),
                atomic(259, "。"),
            ],
            vec![
                VocabEvent::Merge {
                    left: 104,
                    right: 105,
                    result: 256,
                },
                VocabEvent::Merge {
                    left: 256,
                    right: 33,
                    result: 257,
                },
                VocabEvent::Merge {
                    left: 256,
                    right: 120,
                    result: 258,
                },
            ],
            1,
        );
        let tok = Tokenizer::from_model(model).unwrap();
        // '!' is hard punctuation: "hi!" can never merge into 257.
        let ids = tok.encode("hi!").unwrap();
        assert_eq!(ids, vec![256, 33]);
        // 'x' shares a soft boundary: "hix" merges across the pretok split.
        // ("hi" then "x" are actually one ident run, so phase1 sees "hix";
        //  phase1 merges h+i -> 256, then phase2 256+x -> 258)
        let ids = tok.encode("hix").unwrap();
        assert_eq!(ids, vec![258]);
        // '。' hard-splits: "hi。x" -> [256, 259, 'x']
        let ids = tok.encode("hi。x").unwrap();
        assert_eq!(ids, vec![256, 259, 120]);
    }

    #[test]
    fn encode_with_offsets_and_pieces() {
        let model = base_model(vec![atomic(256, "あ")], vec![], 0);
        let tok = Tokenizer::from_model(model).unwrap();
        let spans = tok.encode_with_offsets("aあb").unwrap();
        assert_eq!(spans, vec![(97, 0, 1), (256, 1, 4), (98, 4, 5)]);
        let pieces = tok.tokenize_pieces("aあb").unwrap();
        assert_eq!(pieces, vec!["a", "あ", "b"]);
    }

    #[test]
    fn batch_and_lossy() {
        let tok = Tokenizer::from_model(base_model(vec![], vec![], 0)).unwrap();
        let batch = tok.encode_batch(&["ab", "cd"]).unwrap();
        assert_eq!(batch, vec![vec![97, 98], vec![99, 100]]);
        assert_eq!(tok.decode_lossy(&[97, 98]), "ab");
        assert_eq!(tok.decode_lossy(&[9999]), ""); // unknown ids skipped
    }

    #[test]
    fn kanji_hiragana_cross_helper() {
        assert!(is_kanji_to_hiragana_cross(
            "新".as_bytes(),
            "しい".as_bytes()
        ));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn any_string() -> impl Strategy<Value = String> {
            prop_oneof![
                any::<String>(),
                "[あ-んア-ンー一-龯々〆〇a-zA-Z0-9 .,;:!?()\\[\\]{}+\\-*/=<>\n\r\t　。、・「」]*",
                "[\\u{20000}-\\u{2FA1F}あア🎉🎊x1 ]*",
            ]
        }

        proptest! {
            /// Byte-level model: no atomic chars, no merges -> lossless
            /// decode(encode(s)) == s for arbitrary Unicode.
            #[test]
            fn encode_decode_roundtrip_byte_model(s in any_string()) {
                let tok = Tokenizer::from_model(base_model(vec![], vec![], 0)).unwrap();
                let ids = tok.encode(&s).unwrap();
                prop_assert_eq!(tok.decode(&ids).unwrap(), s);
            }

            /// With atomic chars + a phase1 merge, still lossless.
            #[test]
            fn encode_decode_roundtrip_with_merges(s in any_string()) {
                let model = base_model(
                    vec![
                        atomic(256, "あ"),
                        atomic(257, "。"),
                        merged(258, b"ab", TokenKind::MergedPhase1, true),
                    ],
                    vec![VocabEvent::Merge {
                        left: 97,
                        right: 98,
                        result: 258,
                    }],
                    1,
                );
                let tok = Tokenizer::from_model(model).unwrap();
                let ids = tok.encode(&s).unwrap();
                prop_assert_eq!(tok.decode(&ids).unwrap(), s);
            }
        }
    }
}
