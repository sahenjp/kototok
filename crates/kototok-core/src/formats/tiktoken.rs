//! OpenAI tiktoken mergeable ranks.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};

use super::byte_level::{base64_decode, base64_encode};
use super::{ImportedModel, canonical_tokens, invalid};
use crate::{KototokError, Model, PreTokenizerMode, TokenKind};

const MAX_TIKTOKEN_TOKEN_BYTES: usize = 65_536;
const NIL: usize = usize::MAX;

impl Model {
    /// Loads a tiktoken mergeable-rank table.
    pub fn load_tiktoken(path: impl AsRef<Path>) -> Result<Self, KototokError> {
        from_bytes(&fs::read(path)?)
    }

    /// Reconstructs chronological BPE merges from a tiktoken rank table.
    pub fn from_tiktoken_str(input: &str) -> Result<Self, KototokError> {
        let mut rows = Vec::new();
        let mut ranks = FxHashSet::default();
        let mut tokens = FxHashSet::default();
        for (line_number, line) in input.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let (bytes, rank) = parse_row(line)
                .map_err(|err| invalid(format!("tiktoken line {}: {err}", line_number + 1)))?;
            if bytes.len() > MAX_TIKTOKEN_TOKEN_BYTES {
                return Err(invalid(format!(
                    "tiktoken token at line {} is too large",
                    line_number + 1
                )));
            }
            if !ranks.insert(rank) || !tokens.insert(bytes.clone()) {
                return Err(invalid(format!(
                    "duplicate tiktoken token bytes or rank at line {}",
                    line_number + 1
                )));
            }
            rows.push((bytes, rank));
        }
        rows.sort_unstable_by_key(|&(_, rank)| rank);
        let max_rank = rows
            .last()
            .map(|&(_, rank)| rank)
            .ok_or_else(|| invalid("empty tiktoken rank table"))?;
        let mut builder =
            ImportedModel::new("tiktoken", PreTokenizerMode::StandardGpt, Some(max_rank));
        let mut known = FxHashMap::default();
        for (bytes, rank) in &rows {
            if let &[byte] = bytes.as_slice() {
                let iid = u32::from(byte);
                builder.model.internal_tokens[iid as usize].final_id = Some(*rank);
                known.insert(bytes.clone(), (iid, *rank));
            }
        }
        for byte in 0..=u8::MAX {
            let iid = u32::from(byte);
            if builder.model.internal_tokens[iid as usize]
                .final_id
                .is_none()
            {
                let rank = builder.allocate_final_id()?;
                builder.model.internal_tokens[iid as usize].final_id = Some(rank);
                known.insert(vec![byte], (iid, rank));
            }
        }

        let mut max_known_len = 1usize;
        for (bytes, rank) in rows.into_iter().filter(|(bytes, _)| bytes.len() > 1) {
            if let Some((iid, old_rank)) = known.get_mut(bytes.as_slice()) {
                // A sparse rank table can reference a prefix before naming it.
                // Keep the construction already needed by the earlier token.
                builder.model.internal_tokens[*iid as usize].final_id = Some(rank);
                *old_rank = rank;
                continue;
            }
            let pieces = merge_pieces(&bytes, &known, max_known_len);
            let split = if let [left, right] = pieces.as_slice() {
                Some((known[*left].0, known[*right].0))
            } else {
                (1..bytes.len())
                    .filter(|&i| i <= max_known_len && bytes.len() - i <= max_known_len)
                    .filter_map(|i| {
                        let &(left, left_rank) = known.get(&bytes[..i])?;
                        let &(right, right_rank) = known.get(&bytes[i..])?;
                        Some((left_rank.max(right_rank), left, right))
                    })
                    .min_by_key(|&(rank, _, _)| rank)
                    .map(|(_, left, right)| (left, right))
            };
            if let Some((left, right)) = split {
                let result = builder.push_token(bytes.clone(), TokenKind::MergedPhase1, rank)?;
                builder.push_merge(left, right, result);
                max_known_len = max_known_len.max(bytes.len());
                known.insert(bytes, (result, rank));
            } else {
                synthesize_piece_tree(
                    &mut builder,
                    &bytes,
                    &pieces,
                    &mut known,
                    &mut max_known_len,
                    Some(rank),
                )?;
            }
        }
        builder.finish()
    }

    /// Serializes canonical live tokens as sorted tiktoken mergeable ranks.
    pub fn to_tiktoken_string(&self) -> Result<String, KototokError> {
        let mut tokens = canonical_tokens(self);
        let ranks: FxHashSet<_> = tokens.values().copied().collect();
        if ranks.len() != tokens.len() || tokens.contains_key([].as_slice()) {
            return Err(invalid("empty tiktoken token or conflicting rank"));
        }
        let byte_values: [u8; 256] = std::array::from_fn(|byte| byte as u8);
        let mut next_rank = ranks.iter().max().map_or(0, |&rank| u64::from(rank) + 1);
        for byte in byte_values.as_chunks::<1>().0 {
            if let std::collections::hash_map::Entry::Vacant(entry) = tokens.entry(byte) {
                let rank =
                    u32::try_from(next_rank).map_err(|_| invalid("tiktoken ranks exhausted"))?;
                entry.insert(rank);
                next_rank += 1;
            }
        }
        let mut rows: Vec<_> = tokens.into_iter().collect();
        rows.sort_unstable_by_key(|&(_, rank)| rank);
        let mut output = String::new();
        for (bytes, rank) in rows {
            writeln!(output, "{} {rank}", base64_encode(bytes)).expect("writing to a string");
        }
        Ok(output)
    }

    /// Writes a mergeable-rank table in tiktoken format.
    pub fn save_tiktoken(&self, path: impl AsRef<Path>) -> Result<(), KototokError> {
        fs::write(path, self.to_tiktoken_string()?)?;
        Ok(())
    }
}

pub(super) fn from_bytes(bytes: &[u8]) -> Result<Model, KototokError> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| invalid("tiktoken rank table is not UTF-8"))?;
    Model::from_tiktoken_str(text)
}

pub(super) fn parse_row(line: &str) -> Result<(Vec<u8>, u32), KototokError> {
    let mut fields = line.split_whitespace();
    let token = fields
        .next()
        .ok_or_else(|| invalid("missing tiktoken token"))?;
    let rank = fields
        .next()
        .and_then(|rank| rank.parse().ok())
        .ok_or_else(|| invalid("missing or invalid tiktoken rank"))?;
    if fields.next().is_some() {
        return Err(invalid("too many fields in tiktoken rank row"));
    }
    Ok((base64_decode(token)?, rank))
}

/// Builds a balanced synthetic merge tree for pieces that the sparse rank table
/// never named explicitly. This keeps long-token import at O(L log L) total
/// token bytes instead of the quadratic storage of a left-associated chain.
fn synthesize_piece_tree<'a>(
    builder: &mut ImportedModel,
    span: &'a [u8],
    pieces: &[&'a [u8]],
    known: &mut FxHashMap<Vec<u8>, (u32, u32)>,
    max_known_len: &mut usize,
    root_rank: Option<u32>,
) -> Result<u32, KototokError> {
    if pieces.len() == 1 {
        return Ok(known[pieces[0]].0);
    }
    let mid = pieces.len() / 2;
    let left_len: usize = pieces[..mid].iter().map(|piece| piece.len()).sum();
    let (left_span, right_span) = span.split_at(left_len);
    let (left_pieces, right_pieces) = pieces.split_at(mid);
    let left = synthesize_piece_tree(builder, left_span, left_pieces, known, max_known_len, None)?;
    let right = synthesize_piece_tree(
        builder,
        right_span,
        right_pieces,
        known,
        max_known_len,
        None,
    )?;
    let final_id = match root_rank {
        Some(rank) => rank,
        None => builder.allocate_final_id()?,
    };
    let result = builder.push_token(span.to_vec(), TokenKind::MergedPhase1, final_id)?;
    builder.push_merge(left, right, result);
    *max_known_len = (*max_known_len).max(span.len());
    known.insert(span.to_vec(), (result, final_id));
    Ok(result)
}

#[derive(Clone, Copy)]
struct PieceNode {
    pos: usize,
    prev: usize,
    next: usize,
    active: bool,
}

fn piece_candidate(
    bytes: &[u8],
    nodes: &[PieceNode],
    idx: usize,
    known: &FxHashMap<Vec<u8>, (u32, u32)>,
    max_known_len: usize,
) -> Option<u32> {
    let node = nodes[idx];
    if !node.active || node.prev == NIL || node.next == NIL {
        return None;
    }
    let start = nodes[node.prev].pos;
    let end = nodes[node.next].pos;
    if end - start > max_known_len {
        return None;
    }
    known.get(&bytes[start..end]).map(|&(_, rank)| rank)
}

fn push_piece_candidate(
    bytes: &[u8],
    nodes: &[PieceNode],
    idx: usize,
    known: &FxHashMap<Vec<u8>, (u32, u32)>,
    max_known_len: usize,
    heap: &mut BinaryHeap<Reverse<(u32, usize, usize)>>,
) {
    if idx == NIL {
        return;
    }
    if let Some(rank) = piece_candidate(bytes, nodes, idx, known, max_known_len) {
        heap.push(Reverse((rank, nodes[idx].pos, idx)));
    }
}

/// Reconstructs the minimum-rank binary segmentation with a linked boundary
/// list. Candidates are keyed by the boundary they remove; stale heap entries
/// are rejected by recomputing the candidate at pop time.
fn merge_pieces<'a>(
    bytes: &'a [u8],
    known: &FxHashMap<Vec<u8>, (u32, u32)>,
    max_known_len: usize,
) -> Vec<&'a [u8]> {
    if bytes.len() <= 1 {
        return vec![bytes];
    }
    let mut nodes: Vec<PieceNode> = (0..=bytes.len())
        .map(|pos| PieceNode {
            pos,
            prev: if pos == 0 { NIL } else { pos - 1 },
            next: if pos == bytes.len() { NIL } else { pos + 1 },
            active: true,
        })
        .collect();
    let mut heap = BinaryHeap::new();
    for idx in 1..bytes.len() {
        push_piece_candidate(bytes, &nodes, idx, known, max_known_len, &mut heap);
    }

    while let Some(Reverse((rank, _, idx))) = heap.pop() {
        if piece_candidate(bytes, &nodes, idx, known, max_known_len) != Some(rank) {
            continue;
        }
        let node = nodes[idx];
        let prev = node.prev;
        let next = node.next;
        nodes[idx].active = false;
        nodes[prev].next = next;
        nodes[next].prev = prev;
        push_piece_candidate(bytes, &nodes, prev, known, max_known_len, &mut heap);
        push_piece_candidate(bytes, &nodes, next, known, max_known_len, &mut heap);
    }

    let mut boundaries = Vec::new();
    let mut cur = 0usize;
    while cur != NIL {
        boundaries.push(nodes[cur].pos);
        cur = nodes[cur].next;
    }
    boundaries
        .windows(2)
        .map(|range| &bytes[range[0]..range[1]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tokenizer, VocabEvent};

    #[test]
    fn reconstructs_rank_order_and_synthetic_tokens() {
        let ranked = [
            ("abc", 260),
            ("ab", 257),
            ("cd", 258),
            ("bc", 256),
            ("abcd", 259),
            ("WXYZ", 261),
            ("WX", 262),
            ("a", 15),
            ("b", 3),
            ("c", 11),
            ("d", 5),
            ("日本", 263),
            ("🦀", 264),
        ];
        let mut table = String::new();
        for (text, rank) in ranked {
            writeln!(table, "{} {rank}", base64_encode(text.as_bytes())).unwrap();
        }
        let model = Model::from_tiktoken_str(&table).unwrap();
        let merges: Vec<_> = model
            .events
            .iter()
            .map(|event| {
                let VocabEvent::Merge {
                    left,
                    right,
                    result,
                } = *event
                else {
                    panic!("unexpected prune")
                };
                let bytes = |iid: u32| model.internal_tokens[iid as usize].bytes.as_slice();
                (bytes(left), bytes(right), bytes(result))
            })
            .collect();
        assert_eq!(
            merges[..5],
            [
                (b"b".as_slice(), b"c".as_slice(), b"bc".as_slice()),
                (b"a", b"b", b"ab"),
                (b"c", b"d", b"cd"),
                (b"ab", b"cd", b"abcd"),
                (b"a", b"bc", b"abc"),
            ]
        );
        assert_eq!(
            model.metadata.phase2_start_rank as usize,
            model.events.len()
        );
        assert!(model.internal_tokens.iter().all(|t| t.final_id.is_some()));
        let exported = model.to_tiktoken_string().unwrap();
        let rows: Vec<_> = exported
            .lines()
            .map(|line| parse_row(line).unwrap())
            .collect();
        assert!(rows.windows(2).all(|row| row[0].1 < row[1].1));
        for (text, rank) in ranked {
            assert!(rows.contains(&(text.as_bytes().to_vec(), rank)));
        }
        let tok = Tokenizer::from_model(model).unwrap();
        assert_eq!(tok.encode("abc").unwrap(), [260]);
        assert_eq!(tok.encode("WXYZ").unwrap(), [261]);
        assert_eq!(tok.encode("WX").unwrap(), [262]);
        assert_eq!(tok.encode("日本🦀").unwrap(), [263, 264]);
        for text in [
            "日本語 abc 🦀",
            "fn f() {\n    return 123;\n}",
            "あいうえお",
            "WXY",
        ] {
            assert_eq!(tok.decode(&tok.encode(text).unwrap()).unwrap(), text);
        }
    }

    #[test]
    fn huge_tiktoken_token_import_is_fast() {
        let huge = vec![7u8; 10_000];
        let table = format!("{} 0\n", base64_encode(&huge));
        let start = std::time::Instant::now();
        let model = Model::from_tiktoken_str(&table).unwrap();
        assert!(start.elapsed() < std::time::Duration::from_millis(20));
        assert!(!model.events.is_empty());
    }

    #[test]
    fn rejects_absurdly_large_tiktoken_token() {
        let huge = vec![b'a'; MAX_TIKTOKEN_TOKEN_BYTES + 1];
        let table = format!(
            "{} 0
",
            base64_encode(&huge)
        );
        assert!(matches!(
            Model::from_tiktoken_str(&table),
            Err(KototokError::InvalidFormat(_))
        ));
    }

    #[test]
    fn rejects_invalid_tiktoken_rows_and_ids() {
        for input in [
            "",
            "YQ==",
            "YQ== -1",
            "YQ== 1 extra",
            "YQ== 4294967296",
            "YQ== 4294967295",
            "YQ== 2000000000",
            "Y!== 1",
            "YQ== 1\nYQ== 2",
            "YQ== 1\nYg== 1",
        ] {
            assert!(
                matches!(
                    Model::from_tiktoken_str(input),
                    Err(KototokError::InvalidFormat(_))
                ),
                "{input:?}"
            );
        }
    }
}
