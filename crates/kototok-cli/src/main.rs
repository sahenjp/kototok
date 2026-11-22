//! `kototok` CLI: train / encode / decode / inspect / viz / bench.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Instant;

use clap::{Parser, Subcommand};
use rayon::prelude::*;

use kototok_core::{InternalToken, KototokError, Model, TokenKind, Tokenizer, VocabEvent};
use kototok_train::{TrainPreset, train};

#[derive(Parser)]
#[command(name = "kototok", version, about = "kototok: AsymScript-BPE tokenizer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Train a tokenizer model on a UTF-8 corpus.
    Train {
        /// Corpus text file.
        #[arg(long)]
        input: PathBuf,
        /// Output `.ktok` model path.
        #[arg(long)]
        output: PathBuf,
        /// Optional JSON export of the model.
        #[arg(long)]
        json_out: Option<PathBuf>,
        /// Maximum vocabulary size.
        #[arg(long, default_value_t = 4096)]
        vocab_size: usize,
        /// Ablation preset.
        #[arg(long, default_value = "asym-full", value_parser = parse_preset)]
        preset: TrainPreset,
        /// Override ACSG NPMI exponent (`0` disables ACSG).
        #[arg(long)]
        beta: Option<f64>,
        /// Override PickyBPE IoS threshold.
        #[arg(long)]
        picky_threshold: Option<f64>,
        /// Override Phase-1 vocabulary ratio.
        #[arg(long)]
        phase1_ratio: Option<f64>,
        /// Override Phase-2 minimum NPMI.
        #[arg(long)]
        phase2_min_npmi: Option<f64>,
        /// Override Phase-2 minimum NPMI for pure-ASCII pairs.
        #[arg(long)]
        phase2_min_npmi_ascii: Option<f64>,
        /// Override Phase-2 minimum NPMI for low-gate Kanji->Hiragana
        /// (inflection/okurigana-shaped) pairs.
        #[arg(long)]
        phase2_min_npmi_okuri: Option<f64>,
        /// Override the pure-ASCII share of the Phase-2 vocab budget.
        #[arg(long)]
        phase2_ascii_quota: Option<f64>,
        /// Override minimum char frequency for `AtomicChar`.
        #[arg(long)]
        min_char_freq: Option<u64>,
        /// Override minimum pair frequency.
        #[arg(long)]
        min_pair_freq: Option<u64>,
        /// BEG max NPMI exponent for high-entropy Kanji prefixes (`0` disables).
        #[arg(long)]
        beg_strength: Option<f64>,
        /// BEG right-entropy (nats) below which a Kanji run counts as a stem.
        #[arg(long)]
        beg_stem_max: Option<f64>,
        /// BEG right-entropy (nats) above which a Kanji run counts as a noun.
        #[arg(long)]
        beg_noun_min: Option<f64>,
        /// BEG ln(prefix count) below which a Hiragana run counts as inflection.
        #[arg(long)]
        beg_particle_lo: Option<f64>,
        /// BEG ln(prefix count) above which a Hiragana run counts as particle.
        #[arg(long)]
        beg_particle_hi: Option<f64>,
        /// Apply the BEG gate to Phase-2 merges too (default on, `0` disables).
        #[arg(long)]
        beg_phase2: Option<u8>,
        /// Okurigana boost multiplier for low-gate Kanji->Hiragana pairs.
        #[arg(long)]
        beg_okuri_boost: Option<f64>,
        /// Rayon worker threads (0 = default pool size).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Encode text to token ids / pieces.
    Encode {
        /// Model path (`.ktok` or `.json`).
        #[arg(long)]
        model: PathBuf,
        /// Input text file (stdin when `--text` is absent too).
        #[arg(long)]
        input: Option<PathBuf>,
        /// Literal text to encode.
        #[arg(long)]
        text: Option<String>,
        /// Print space-separated ids.
        #[arg(long)]
        ids: bool,
        /// Print `|`-separated pieces.
        #[arg(long)]
        pieces: bool,
        /// Emit the eval JSON schema.
        #[arg(long)]
        json: bool,
    },
    /// Decode comma-separated final token ids.
    Decode {
        #[arg(long)]
        model: PathBuf,
        /// e.g. `--ids 104,105,256`.
        #[arg(long)]
        ids: String,
    },
    /// Print the model's UTF-8 audit report.
    Inspect {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Visualize tokenization pieces for a text.
    Viz {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        text: String,
    },
    /// Boundary-aligned merge surgery: prune learned merges that cross gold
    /// morphological boundaries (Aligned BPE-Knockout).
    Refine {
        /// Model path (`.ktok` or `.json`).
        #[arg(long)]
        model: PathBuf,
        /// Gold boundary JSONL (`text`, `byte_spans`, `okurigana_words`).
        #[arg(long)]
        gold: PathBuf,
        /// Output refined `.ktok` path.
        #[arg(long)]
        output: PathBuf,
        /// Minimum boundary violations before pruning a merge.
        #[arg(long, default_value_t = 1)]
        min_violations: usize,
        /// Minimum occurrences of an under-merged gold token before grafting
        /// new merge events that produce it (`0` disables grafting).
        #[arg(long, default_value_t = 1)]
        min_graft_freq: usize,
    },
    /// Time `encode` throughput on an input file.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        /// Parallel chunk encoding workers (1 = single-threaded).
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Timed iterations after one warmup.
        #[arg(long, default_value_t = 5)]
        iters: usize,
        #[arg(long)]
        json: bool,
    },
}

fn parse_preset(s: &str) -> Result<TrainPreset, String> {
    TrainPreset::from_str(s)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type DynError = Box<dyn std::error::Error>;

fn run() -> Result<(), DynError> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train {
            input,
            output,
            json_out,
            vocab_size,
            preset,
            beta,
            picky_threshold,
            phase1_ratio,
            phase2_min_npmi,
            phase2_min_npmi_ascii,
            phase2_min_npmi_okuri,
            phase2_ascii_quota,
            min_char_freq,
            min_pair_freq,
            beg_strength,
            beg_stem_max,
            beg_noun_min,
            beg_particle_lo,
            beg_particle_hi,
            beg_phase2,
            beg_okuri_boost,
            threads,
        } => {
            if let Some(t) = threads
                && t > 0
            {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build_global()
                    .ok();
            }
            let bytes = fs::read(&input)?;
            let corpus = String::from_utf8_lossy(&bytes);
            let mut cfg = preset.config(vocab_size);
            if let Some(v) = beta {
                cfg.acsg_npmi_beta = v;
            }
            if let Some(v) = picky_threshold {
                cfg.picky_ios_threshold = v;
            }
            if let Some(v) = phase1_ratio {
                cfg.phase1_vocab_ratio = v;
            }
            if let Some(v) = phase2_min_npmi {
                cfg.phase2_min_npmi = v;
            }
            if let Some(v) = phase2_min_npmi_ascii {
                cfg.phase2_min_npmi_ascii = v;
            }
            if let Some(v) = phase2_min_npmi_okuri {
                cfg.phase2_min_npmi_okuri = v;
            }
            if let Some(v) = phase2_ascii_quota {
                cfg.phase2_ascii_quota = v;
            }
            if let Some(v) = min_char_freq {
                cfg.min_char_freq = v;
            }
            if let Some(v) = min_pair_freq {
                cfg.min_pair_freq = v;
            }
            if let Some(v) = beg_strength {
                cfg.beg_strength = v;
            }
            if let Some(v) = beg_stem_max {
                cfg.beg_stem_max = v;
            }
            if let Some(v) = beg_noun_min {
                cfg.beg_noun_min = v;
            }
            if let Some(v) = beg_particle_lo {
                cfg.beg_particle_lo = v;
            }
            if let Some(v) = beg_particle_hi {
                cfg.beg_particle_hi = v;
            }
            if let Some(v) = beg_phase2 {
                cfg.beg_phase2 = v != 0;
            }
            if let Some(v) = beg_okuri_boost {
                cfg.beg_okuri_boost = v;
            }
            let model = train(&corpus, &cfg)?;
            model.save_binary(&output)?;
            if let Some(path) = &json_out {
                model.save_json(path)?;
            }
            let audit = model.audit_utf8_validity();
            eprintln!(
                "trained '{}': alive={} pruned={} events={} phase1_vocab={} phase2_rank={} invalid_utf8={}",
                cfg.name,
                audit.alive_vocab_size,
                audit.pruned_tokens,
                model.events.len(),
                model.metadata.phase1_vocab_size,
                model.metadata.phase2_start_rank,
                audit.invalid_utf8_learned_tokens,
            );
            Ok(())
        }
        Command::Encode {
            model,
            input,
            text,
            ids,
            pieces,
            json,
        } => {
            let tok = load_tokenizer(&model)?;
            let text = read_text(input.as_deref(), text)?;
            if json {
                let spans = tok.encode_with_offsets(&text)?;
                let ids: Vec<u32> = spans.iter().map(|s| s.0).collect();
                let byte_lens: Vec<usize> = spans.iter().map(|s| s.2 - s.1).collect();
                let mut pieces = Vec::with_capacity(ids.len());
                for &id in &ids {
                    pieces.push(String::from_utf8_lossy(&tok.decode_bytes(&[id])?).into_owned());
                }
                println!(
                    "{}",
                    serde_json::json!({
                        "ids": ids,
                        "pieces": pieces,
                        "byte_lens": byte_lens,
                        "token_count": ids.len(),
                        "byte_count": text.len(),
                        "char_count": text.chars().count(),
                    })
                );
            } else if pieces {
                println!("{}", tok.tokenize_pieces(&text)?.join("|"));
            } else {
                let _ = ids; // `--ids` is the default human output.
                let out: Vec<String> = tok.encode(&text)?.iter().map(u32::to_string).collect();
                println!("{}", out.join(" "));
            }
            Ok(())
        }
        Command::Decode { model, ids } => {
            let tok = load_tokenizer(&model)?;
            let ids: Vec<u32> = ids
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<u32>())
                .collect::<Result<_, _>>()?;
            println!("{}", tok.decode(&ids)?);
            Ok(())
        }
        Command::Inspect { model, json } => {
            let model = Model::load_auto(&model)?;
            let audit = model.audit_utf8_validity();
            if json {
                println!("{}", serde_json::to_string_pretty(&audit)?);
            } else {
                println!("name: {}", model.metadata.name);
                println!("events: {}", model.events.len());
                println!("{audit:#?}");
            }
            Ok(())
        }
        Command::Viz { model, text } => {
            let tok = load_tokenizer(&model)?;
            let pieces = tok.tokenize_pieces(&text)?;
            let chars = text.chars().count();
            println!("{}", pieces.join("|"));
            println!("tokens: {}", pieces.len());
            println!("chars: {chars}");
            println!(
                "chars/token: {:.4}",
                if pieces.is_empty() {
                    0.0
                } else {
                    chars as f64 / pieces.len() as f64
                }
            );
            Ok(())
        }
        Command::Refine {
            model,
            gold,
            output,
            min_violations,
            min_graft_freq,
        } => refine_model(&model, &gold, &output, min_violations, min_graft_freq),
        Command::Bench {
            model,
            input,
            threads,
            iters,
            json,
        } => {
            if threads > 1 {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global()
                    .ok();
            }
            let tok = load_tokenizer(&model)?;
            let text = fs::read_to_string(&input)?;

            // Warmup (also produces the reference token count).
            let tokens = tok.encode(&text)?.len();

            let start = Instant::now();
            if threads > 1 {
                let blocks = line_blocks(&text, threads * 8);
                for _ in 0..iters.max(1) {
                    let lens: Vec<usize> = blocks
                        .par_iter()
                        .map(|b| tok.encode(b).map(|v| v.len()))
                        .collect::<Result<_, KototokError>>()?;
                    std::hint::black_box(lens);
                }
            } else {
                for _ in 0..iters.max(1) {
                    std::hint::black_box(tok.encode(&text)?);
                }
            }
            let total_secs = start.elapsed().as_secs_f64();
            let iters = iters.max(1);
            let mb_per_sec = text.len() as f64 * iters as f64 / total_secs / 1e6;
            let tokens_per_sec = tokens as f64 * iters as f64 / total_secs;
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "threads": threads,
                        "bytes": text.len(),
                        "tokens": tokens,
                        "iters": iters,
                        "total_secs": total_secs,
                        "mb_per_sec": mb_per_sec,
                        "tokens_per_sec": tokens_per_sec,
                    })
                );
            } else {
                println!("threads: {threads}");
                println!("bytes: {}", text.len());
                println!("tokens: {tokens}");
                println!("iters: {iters}");
                println!("total_secs: {total_secs:.4}");
                println!("mb_per_sec: {mb_per_sec:.2}");
                println!("tokens_per_sec: {tokens_per_sec:.0}");
            }
            Ok(())
        }
    }
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, KototokError> {
    Tokenizer::from_model(Model::load_auto(path)?)
}

fn read_text(input: Option<&Path>, text: Option<String>) -> io::Result<String> {
    if let Some(t) = text {
        return Ok(t);
    }
    if let Some(p) = input {
        return fs::read_to_string(p);
    }
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

/// Boundary-aligned bidirectional merge surgery (Aligned BPE-Knockout).
///
/// Phase A (prune): attributes every predicted piece that strictly contains
/// a gold internal boundary to the merge event that produced it, and
/// inserts a `Prune` event right after that merge.
/// Phase B (graft): finds gold tokens the model under-merges (predicted
/// pieces strictly inside a gold token span) and appends merge events that
/// produce the whole word — a post-hoc vocabulary *repair*, not just
/// deletion. Post-hoc: no retraining needed.
fn refine_model(
    model_path: &Path,
    gold_path: &Path,
    out_path: &Path,
    min_violations: usize,
    min_graft_freq: usize,
) -> Result<(), DynError> {
    let mut model = Model::load_auto(model_path)?;
    let tok = Tokenizer::from_model(model.clone())?;

    let mut fid_to_internal = std::collections::HashMap::new();
    for t in &model.internal_tokens {
        if let Some(fid) = t.final_id {
            fid_to_internal.insert(fid, t.internal_id);
        }
    }
    let mut internal_to_merge: std::collections::HashMap<u32, (usize, u32, u32)> =
        std::collections::HashMap::new();
    for (i, ev) in model.events.iter().enumerate() {
        if let VocabEvent::Merge {
            left,
            right,
            result,
        } = ev
        {
            internal_to_merge.insert(*result, (i, *left, *right));
        }
    }

    // internal_id -> (occurrences, violations)
    let mut stats: std::collections::HashMap<u32, (usize, usize)> =
        std::collections::HashMap::new();
    // token bytes -> (frequency, internal-id chain producing it)
    let mut graft_map: std::collections::HashMap<Vec<u8>, (usize, Vec<u32>)> =
        std::collections::HashMap::new();
    let mut n_lines = 0usize;
    for line in fs::read_to_string(gold_path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let Some(text) = v["text"].as_str() else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        n_lines += 1;
        let text_len = text.len();

        let gold_tokens: Vec<(usize, usize)> = v["byte_spans"]
            .as_array()
            .map(|spans| {
                spans
                    .iter()
                    .filter_map(|s| {
                        Some((s[0].as_u64()? as usize, s[1].as_u64()? as usize))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut gbounds: Vec<usize> = v["byte_spans"]
            .as_array()
            .map(|spans| {
                spans
                    .iter()
                    .flat_map(|s| {
                        [s[0].as_u64().unwrap_or(0), s[1].as_u64().unwrap_or(0)]
                    })
                    .map(|b| b as usize)
                    .filter(|&b| b > 0 && b < text_len)
                    .collect()
            })
            .unwrap_or_default();
        gbounds.sort_unstable();
        gbounds.dedup();

        // Boundaries inside okurigana words are protected: pruning those
        // merges would split the words and break okurigana integrity.
        let oku_spans: Vec<(usize, usize)> = v["okurigana_words"]
            .as_array()
            .map(|ws| {
                ws.iter()
                    .filter_map(|w| {
                        let sp = w["byte_span"].as_array()?;
                        Some((sp[0].as_u64()? as usize, sp[1].as_u64()? as usize))
                    })
                    .collect()
            })
            .unwrap_or_default();
        gbounds.retain(|b| {
            !oku_spans
                .iter()
                .any(|&(ws, we)| ws < *b && *b < we)
        });

        let Ok(pieces) = tok.encode_with_offsets(text) else {
            continue;
        };
        for (fid, s, e) in &pieces {
            let Some(&iid) = fid_to_internal.get(fid) else {
                continue;
            };
            let ent = stats.entry(iid).or_default();
            ent.0 += 1;
            // Over-merge: the piece strictly contains a gold boundary. Under-
            // merging (splitting inside oku words) is NOT prunable — it needs
            // more merges, not fewer.
            let crosses = gbounds.iter().any(|&b| *s < b && b < *e);
            if crosses {
                ent.1 += 1;
            }
        }

        // Phase B collection: gold tokens the model under-merges — covered
        // by >=2 predicted pieces that tile the span exactly.
        if min_graft_freq > 0 {
            for &(gs, ge) in &gold_tokens {
                let mut expect = gs;
                let mut inside: Vec<u32> = Vec::new();
                let mut tiled = true;
                for (fid, s, e) in &pieces {
                    if *s < gs || *e > ge {
                        continue;
                    }
                    if *s != expect {
                        tiled = false;
                        break;
                    }
                    expect = *e;
                    match fid_to_internal.get(fid) {
                        Some(&iid) => inside.push(iid),
                        None => {
                            tiled = false;
                            break;
                        }
                    }
                }
                if tiled && expect == ge && inside.len() >= 2 {
                    let key = text.as_bytes()[gs..ge].to_vec();
                    let ent = graft_map.entry(key).or_insert((0, inside));
                    ent.0 += 1;
                }
            }
        }
    }

    let mut inserts: Vec<(usize, VocabEvent)> = Vec::new();
    for (&iid, &(_occ, viol)) in &stats {
        if viol < min_violations {
            continue;
        }
        let Some(&(rank, left, right)) = internal_to_merge.get(&iid) else {
            continue;
        };
        let child_alive = |id: u32| {
            model
                .internal_tokens
                .iter()
                .find(|t| t.internal_id == id)
                .and_then(|t| t.final_id)
                .is_some()
        };
        if !(child_alive(left) && child_alive(right)) {
            continue;
        }
        inserts.push((
            rank + 1,
            VocabEvent::Prune {
                target: iid,
                split_into: vec![left, right],
            },
        ));
    }
    inserts.sort_by_key(|(idx, _)| std::cmp::Reverse(*idx));

    let pruned_targets: std::collections::HashSet<u32> = inserts
        .iter()
        .filter_map(|(_, ev)| {
            if let VocabEvent::Prune { target, .. } = ev {
                Some(*target)
            } else {
                None
            }
        })
        .collect();

    let phase2 = model.metadata.phase2_start_rank as usize;
    let before_phase2 = inserts.iter().filter(|(idx, _)| *idx <= phase2).count();
    let n_pruned = inserts.len();
    for (idx, ev) in inserts {
        model.events.insert(idx, ev);
    }
    model.metadata.phase2_start_rank += before_phase2 as u32;
    model.metadata.pruned_count += n_pruned;

    // Phase B (graft): append merge events producing under-merged gold
    // tokens. Appended at the end so they only fire where nothing else
    // merges — surgical, never competing with learned vocab.
    let mut pair_to_result: std::collections::HashMap<(u32, u32), u32> =
        std::collections::HashMap::new();
    for ev in &model.events {
        if let VocabEvent::Merge {
            left,
            right,
            result,
        } = ev
        {
            pair_to_result.insert((*left, *right), *result);
        }
    }
    let mut next_iid = model.internal_tokens.len() as u32;
    let mut next_fid = model
        .internal_tokens
        .iter()
        .filter_map(|t| t.final_id)
        .max()
        .map_or(0, |m| m + 1);
    let mut graft_list: Vec<_> = graft_map.into_iter().collect();
    graft_list.sort_by_key(|(_, (f, _))| std::cmp::Reverse(*f));
    let mut n_grafts = 0usize;
    for (_bytes, (freq, chain)) in graft_list {
        if freq < min_graft_freq || chain.len() < 2 {
            continue;
        }
        if chain.iter().any(|c| pruned_targets.contains(c)) {
            continue;
        }
        let mut cur = chain[0];
        for &r in &chain[1..] {
            if let Some(&res) = pair_to_result.get(&(cur, r)) {
                cur = res;
                continue;
            }
            let bytes = [
                model.internal_tokens[cur as usize].bytes.as_slice(),
                model.internal_tokens[r as usize].bytes.as_slice(),
            ]
            .concat();
            model.internal_tokens.push(InternalToken {
                internal_id: next_iid,
                bytes,
                kind: TokenKind::MergedPhase2,
                // Real final_id: a partial chain may surface this token in
                // other contexts, and emitting it must be legal.
                final_id: Some(next_fid),
            });
            pair_to_result.insert((cur, r), next_iid);
            model.events.push(VocabEvent::Merge {
                left: cur,
                right: r,
                result: next_iid,
            });
            cur = next_iid;
            next_iid += 1;
            next_fid += 1;
            n_grafts += 1;
        }
    }
    model.metadata.vocab_size = next_fid as usize;

    model.metadata.name = format!("{}-refined", model.metadata.name);
    model.metadata.train_notes = format!(
        "aligned-BKO: {n_pruned} prunes + {n_grafts} grafts via {n_lines} gold sentences"
    );
    model.save_binary(out_path)?;
    println!(
        "refined: {n_pruned} prunes + {n_grafts} grafts ({n_lines} gold lines) -> {}",
        out_path.display()
    );
    Ok(())
}

/// Line-aligned, char-boundary-safe block splitter for parallel bench.
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
