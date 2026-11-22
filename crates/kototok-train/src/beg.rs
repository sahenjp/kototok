//! Boundary-Entropy Gate (BEG): dictionary-free morphological boundary
//! inference for `Kanji -> Hiragana` merge scoring.
//!
//! Both desirable merges (verb/adjective stem + okurigana, e.g. `食 + べる`)
//! and undesirable merges (noun + particle, e.g. `日本 + の`) look identical to
//! ACSG: a `Kanji -> Hiragana` byte boundary cross. BEG distinguishes them by
//! the right-branching entropy of the trailing Kanji run measured over the
//! corpus:
//!
//! - Stems (`食`, `新し`) are followed by a small, closed set of inflectional
//!   continuations (`べる`, `べた`, `い`, `く`) -> low entropy.
//! - Complete nouns (`日本`, `名詞`) are followed by a wide open set of
//!   particles and auxiliaries (`の`, `は`, `が`, `を`, `に`...) -> high
//!   entropy.
//!
//! The gate scales the NPMI penalty exponent between `beg_stem_max` (treated
//! as a stem: merge freely) and `beg_noun_min` (treated as a noun: full
//! `beg_strength` penalty), keeping the dictionary-free property: every
//! decision is corpus-statistical.

use rustc_hash::FxHashMap;

use kototok_core::{ScriptClass, classify_char};

use crate::counter::CorpusCounts;

/// Two-sided BEG score tables.
pub struct BegTables {
    /// Kanji suffix -> gate `0.0` (stem) .. `1.0` (complete noun), from the
    /// prefix's right-branching entropy over following Hiragana runs.
    pub suffix: FxHashMap<Box<[u8]>, f64>,
    /// Hiragana run -> gate `0.0` (inflection) .. `1.0` (particle), from the
    /// log count of distinct Kanji prefixes it attaches to. Catches
    /// noun+particle merges whose noun has a degenerate (single-follower)
    /// distribution, e.g. `古代教会` + `の`.
    pub hira: FxHashMap<Box<[u8]>, f64>,
}

/// Returns the UTF-8 suffix of `bytes` formed by the trailing run of `Kanji`
/// characters, or an empty slice if `bytes` does not end in `Kanji`.
pub fn trailing_kanji_run(bytes: &[u8]) -> &[u8] {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return &[];
    };
    let mut run_start = s.len();
    let mut prev_kanji = false;
    for (i, ch) in s.char_indices() {
        let is_kanji = classify_char(ch) == ScriptClass::Kanji;
        if is_kanji && !prev_kanji {
            run_start = i;
        }
        if !is_kanji {
            run_start = s.len();
        }
        prev_kanji = is_kanji;
    }
    &bytes[run_start..]
}

/// Builds `kanji_suffix -> normalized gate (0.0..=1.0)` scores.
///
/// Raw right-branching entropy `H(prefix)` (nats) over the following
/// Hiragana-run distribution is mapped through
/// `clamp((H - stem_max) / (noun_min - stem_max), 0, 1)`. Every char-aligned
/// suffix of each observed Kanji prefix inherits its score so that partially
/// merged left operands (e.g. `詞` inside `名詞`) resolve; on collisions the
/// maximum (most noun-like) score wins.
/// Returns the UTF-8 prefix of `bytes` formed by the leading run of
/// `Hiragana` characters, or an empty slice if it does not start with
/// `Hiragana`.
pub fn leading_hiragana_run(bytes: &[u8]) -> &[u8] {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return &[];
    };
    let mut end = 0;
    for (i, ch) in s.char_indices() {
        if classify_char(ch) == ScriptClass::Hiragana {
            end = i + ch.len_utf8();
        } else {
            break;
        }
    }
    &bytes[..end]
}

/// Returns the UTF-8 suffix of `bytes` formed by the trailing run of
/// `Hiragana` characters, or an empty slice if it does not end in `Hiragana`.
pub fn trailing_hiragana_run(bytes: &[u8]) -> &[u8] {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return &[];
    };
    let mut start = s.len();
    for (i, ch) in s.char_indices().rev() {
        if classify_char(ch) == ScriptClass::Hiragana {
            start = i;
        } else {
            break;
        }
    }
    &bytes[start..]
}

/// True when `bytes` is a non-empty all-Hiragana string.
pub fn all_hiragana(bytes: &[u8]) -> bool {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return false;
    };
    !s.is_empty() && s.chars().all(|c| classify_char(c) == ScriptClass::Hiragana)
}

/// True when `bytes` is made only of `Kanji` and `Hiragana` and contains at
/// least one `Kanji` — the shape of a stem-plus-okurigana word or prefix.
pub fn is_kanji_hiragana_word(bytes: &[u8]) -> bool {
    let Ok(s) = std::str::from_utf8(bytes) else {
        return false;
    };
    let mut has_kanji = false;
    let ok = s.chars().all(|c| {
        let cl = classify_char(c);
        has_kanji |= cl == ScriptClass::Kanji;
        cl == ScriptClass::Kanji || cl == ScriptClass::Hiragana
    });
    ok && has_kanji
}

pub fn build_beg_scores(
    counts: &CorpusCounts,
    stem_max: f64,
    noun_min: f64,
    particle_lo: f64,
    particle_hi: f64,
) -> BegTables {
    let mut grouped: FxHashMap<&[u8], Vec<(&[u8], u64)>> = FxHashMap::default();
    let mut hira_prefixes: FxHashMap<&[u8], u64> = FxHashMap::default();
    for ((prefix, hira), &c) in &counts.k2h_counts {
        grouped
            .entry(prefix.as_ref())
            .or_default()
            .push((hira.as_ref(), c));
        *hira_prefixes.entry(hira.as_ref()).or_default() += 1;
    }

    let p_span = (particle_hi - particle_lo).max(f64::EPSILON);
    let hira = hira_prefixes
        .into_iter()
        .map(|(h, n)| {
            let g = ((n as f64).ln() - particle_lo) / p_span;
            (Box::from(h), g.clamp(0.0, 1.0))
        })
        .collect();

    let span = (noun_min - stem_max).max(f64::EPSILON);
    let mut out: FxHashMap<Box<[u8]>, f64> = FxHashMap::default();
    for (prefix, followers) in grouped {
        let total: u64 = followers.iter().map(|f| f.1).sum();
        let h = if total == 0 {
            0.0
        } else {
            -followers
                .iter()
                .map(|&(_, c)| {
                    let p = c as f64 / total as f64;
                    p * p.ln()
                })
                .sum::<f64>()
        };
        let gate = ((h - stem_max) / span).clamp(0.0, 1.0);

        let Ok(prefix_str) = std::str::from_utf8(prefix) else {
            continue;
        };
        for (i, _) in prefix_str.char_indices() {
            if i == 0 {
                continue;
            }
            let suffix = &prefix[i..];
            match out.get_mut(suffix) {
                Some(v) => *v = v.max(gate),
                None => {
                    out.insert(Box::from(suffix), gate);
                }
            }
        }
        match out.get_mut(prefix) {
            Some(v) => *v = v.max(gate),
            None => {
                out.insert(Box::from(prefix), gate);
            }
        }
    }
    BegTables { suffix: out, hira }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts_with(pairs: &[(&[u8], &[u8], u64)]) -> CorpusCounts {
        let mut c = CorpusCounts::default();
        for &(p, h, n) in pairs {
            *c.k2h_counts
                .entry((Box::from(p), Box::from(h)))
                .or_default() += n;
        }
        c
    }

    #[test]
    fn trailing_run_extracts_kanji_suffix() {
        assert_eq!(trailing_kanji_run("食".as_bytes()), "食".as_bytes());
        assert_eq!(trailing_kanji_run("名詞".as_bytes()), "名詞".as_bytes());
        assert_eq!(trailing_kanji_run("abc食".as_bytes()), "食".as_bytes());
        assert_eq!(trailing_kanji_run("食べ".as_bytes()), "".as_bytes());
        assert_eq!(trailing_kanji_run("".as_bytes()), "".as_bytes());
    }

    #[test]
    fn stems_score_lower_than_nouns() {
        // Stem 食 is followed only by べる/べた/べて (closed inflections).
        // Noun 日本 is followed by a wide particle set.
        let c = counts_with(&[
            ("食".as_bytes(), "べる".as_bytes(), 50),
            ("食".as_bytes(), "べた".as_bytes(), 30),
            ("食".as_bytes(), "べて".as_bytes(), 20),
            ("日本".as_bytes(), "の".as_bytes(), 20),
            ("日本".as_bytes(), "は".as_bytes(), 18),
            ("日本".as_bytes(), "が".as_bytes(), 15),
            ("日本".as_bytes(), "を".as_bytes(), 12),
            ("日本".as_bytes(), "に".as_bytes(), 12),
            ("日本".as_bytes(), "で".as_bytes(), 10),
            ("日本".as_bytes(), "から".as_bytes(), 7),
            ("日本".as_bytes(), "と".as_bytes(), 6),
        ]);
        let m = build_beg_scores(&c, 0.7, 1.8, 6.0, 9.0);
        let stem = *m.suffix.get("食".as_bytes()).unwrap();
        let noun = *m.suffix.get("日本".as_bytes()).unwrap();
        assert!(stem < 0.5, "stem gate {stem}");
        assert!(noun > 0.9, "noun gate {noun}");
    }

    #[test]
    fn suffix_keys_inherit_prefix_score() {
        let c = counts_with(&[("名詞".as_bytes(), "の".as_bytes(), 10)]);
        let m = build_beg_scores(&c, 0.0, 1.0, 6.0, 9.0);
        assert_eq!(
            m.suffix.get("詞".as_bytes()),
            m.suffix.get("名詞".as_bytes())
        );
    }

    #[test]
    fn particle_runs_gate_higher_than_inflections() {
        let mut pairs = vec![];
        // `の` attaches to 5000 distinct nouns; `べる` to 5 stems.
        for i in 0..5000u32 {
            pairs.push((format!("名{i}").leak().as_bytes(), "の".as_bytes(), 3u64));
        }
        for stem in ["食", "調", "比", "並", "選"] {
            pairs.push((stem.as_bytes(), "べる".as_bytes(), 100));
        }
        let c = counts_with(&pairs);
        let m = build_beg_scores(&c, 0.7, 1.8, 6.0, 9.0);
        let particle = *m.hira.get("の".as_bytes()).unwrap();
        let inflection = *m.hira.get("べる".as_bytes()).unwrap();
        // ln(5000) = 8.52 -> gate (8.52 - 6) / 3 = 0.84
        assert!(particle > 0.8, "particle gate {particle}");
        assert_eq!(inflection, 0.0, "inflection gate {inflection}");
    }

    #[test]
    fn leading_run_extracts_hiragana_prefix() {
        assert_eq!(leading_hiragana_run("べる".as_bytes()), "べる".as_bytes());
        assert_eq!(leading_hiragana_run("の本".as_bytes()), "の".as_bytes());
        assert_eq!(leading_hiragana_run("食".as_bytes()), "".as_bytes());
    }
}
