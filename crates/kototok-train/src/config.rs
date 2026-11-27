//! Training configuration and the six ablation presets.

use std::fmt;
use std::str::FromStr;

use kototok_core::{ByteMode, PreTokenizerConfig, PreTokenizerMode};

/// Full training configuration for [`crate::train`].
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Model name recorded in `ModelMetadata::name`.
    pub name: String,
    /// Maximum number of alive tokens in the exported vocabulary.
    pub vocab_size: usize,
    /// DFA pre-tokenizer and byte-mode configuration.
    pub pretok: PreTokenizerConfig,
    /// Minimum corpus frequency for a non-ASCII `char` to become `AtomicChar`.
    pub min_char_freq: u64,
    /// Maximum number of atomic non-ASCII characters admitted during vocabulary init.
    pub max_atomic_chars: Option<usize>,
    /// Seed core Unicode symbol, emoji, flag, math, and combining-mark blocks as
    /// guaranteed `AtomicChar` tokens when `vocab_size` has sufficient budget.
    pub seed_unicode_symbols: bool,
    /// Minimum adjacent pair frequency required for a merge candidate.
    pub min_pair_freq: u64,
    /// ACSG NPMI penalty exponent for `Kanji -> Hiragana` cross pairs (`0.0` disables).
    pub acsg_npmi_beta: f64,
    /// Enable PickyBPE IoS pruning of `MergedPhase1` intermediate tokens.
    pub picky_enabled: bool,
    /// IoS threshold at/above which a merged constituent is pruned.
    pub picky_ios_threshold: f64,
    /// Enable Phase-2 soft-boundary superword merges.
    pub phase2_enabled: bool,
    /// Fraction of `vocab_size` filled by Phase 1 when Phase 2 is enabled.
    pub phase1_vocab_ratio: f64,
    /// Minimum NPMI a Phase-2 candidate pair must reach.
    pub phase2_min_npmi: f64,
    /// Looser Phase-2 NPMI floor for pairs made of ASCII bytes only.
    /// Pure-Latin superwords carry no Japanese morphology risk, so they may
    /// merge at a much lower NPMI than JA-script pairs. Set equal to
    /// `phase2_min_npmi` to disable the split floor.
    pub phase2_min_npmi_ascii: f64,
    /// Looser Phase-2 NPMI floor for `Kanji -> Hiragana` pairs whose right
    /// side is an inflectional run (BEG gate_R < 0.9): stem + okurigana
    /// shapes merge almost as freely as ASCII, while particle-like runs
    /// (gate_R >= 0.95) stay hard-vetoed and the mid band keeps the normal
    /// floor. Set equal to `phase2_min_npmi` to disable.
    pub phase2_min_npmi_okuri: f64,
    /// Maximum fraction of the Phase-2 vocabulary budget that pure-ASCII
    /// merges may occupy. `1.0` disables the quota.
    pub phase2_ascii_quota: f64,
    /// Use the loose ASCII NPMI floor for every Phase-2 pair containing no Japanese script.
    pub phase2_non_ja_loose: bool,
    /// Minimum corpus frequency for a Phase-2 segment to participate in training.
    pub min_segment_freq: u64,
    /// Maximum number of chunks per Phase-2 training segment window (1..=16).
    /// Smaller windows (e.g. 2-3) let repeated natural-language bigrams/trigrams
    /// survive `min_segment_freq >= 2` without blowing up memory.
    pub phase2_max_segment_chunks: usize,
    /// When `min_segment_freq > 1`, decompose dropped low-frequency segments
    /// into disjoint sub-windows of this length (`0` disables fallback).
    pub phase2_fallback_window: usize,
    /// BEG strength: maximum NPMI exponent applied to `Kanji -> Hiragana`
    /// merges whose left Kanji run is a high-entropy (complete-noun) prefix.
    /// `0.0` disables the gate entirely.
    pub beg_strength: f64,
    /// Right-branching entropy (nats) at/below which the Kanji run is treated
    /// as an inflectional stem -> merge free of penalty.
    pub beg_stem_max: f64,
    /// Right-branching entropy (nats) at/above which the Kanji run is treated
    /// as a complete noun -> full `beg_strength` penalty.
    pub beg_noun_min: f64,
    /// ln(#distinct Kanji prefixes) below which a Hiragana run counts as an
    /// inflectional continuation -> no extra penalty.
    pub beg_particle_lo: f64,
    /// ln(#distinct Kanji prefixes) above which a Hiragana run counts as a
    /// particle -> full `beg_strength` penalty.
    pub beg_particle_hi: f64,
    /// Also apply the BEG gate to Phase-2 (soft-boundary superword) merges.
    pub beg_phase2: bool,
    /// Okurigana boost: `Kanji -> Hiragana` pairs with low combined gate
    /// (stem + inflection shape) get score `x(1 + boost*(1-gate))`.
    /// `0.0` disables the bonus.
    pub beg_okuri_boost: f64,
    /// Extra never-mergeable special tokens appended after the byte entries.
    pub special_tokens: Vec<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainPreset::AsymFull.config(4096)
    }
}

/// The six ablation presets compared by `eval/run_eval.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrainPreset {
    /// Full AsymScript-BPE: ACSG + PickyBPE + Phase 2.
    AsymFull,
    /// AsymScript + ACSG + PickyBPE, no Phase 2.
    AsymNoPhase2,
    /// AsymScript + ACSG + Phase 2, no PickyBPE pruning.
    AsymNoPicky,
    /// AsymScript + PickyBPE + Phase 2, no ACSG scoring.
    AsymNoAcsg,
    /// Strict script boundaries, raw frequency BPE only.
    StrictScript,
    /// Standard GPT-style byte-level BPE baseline.
    StandardBbpe,
    /// Multilingual + math + code global preset (not part of the six ablations).
    Global,
}

impl TrainPreset {
    /// All presets in evaluation order.
    pub const ALL: [Self; 6] = [
        Self::AsymFull,
        Self::AsymNoPhase2,
        Self::AsymNoPicky,
        Self::AsymNoAcsg,
        Self::StrictScript,
        Self::StandardBbpe,
    ];

    /// Canonical CLI name (matches `eval/run_eval.py`).
    pub fn name(self) -> &'static str {
        match self {
            Self::AsymFull => "asym-full",
            Self::AsymNoPhase2 => "asym-no-phase2",
            Self::AsymNoPicky => "asym-no-picky",
            Self::AsymNoAcsg => "asym-no-acsg",
            Self::StrictScript => "strict-script",
            Self::StandardBbpe => "standard-bbpe",
            Self::Global => "global",
        }
    }

    /// Builds the `TrainConfig` for this preset at `vocab_size`.
    pub fn config(self, vocab_size: usize) -> TrainConfig {
        if self == Self::Global {
            return TrainConfig {
                name: self.name().to_string(),
                vocab_size,
                pretok: PreTokenizerConfig {
                    mode: PreTokenizerMode::AsymScript,
                    byte_mode: ByteMode::Utf8Atomic,
                    attach_ascii_leading_space: true,
                    max_digit_run: 3,
                    max_chunk_chars: 128,
                    max_segment_chunks: 64,
                    multilingual_scripts: true,
                    attach_space_to_punct: true,
                },
                min_char_freq: 5,
                max_atomic_chars: Some(18_000),
                seed_unicode_symbols: true,
                min_pair_freq: 2,
                acsg_npmi_beta: 0.0,
                picky_enabled: true,
                picky_ios_threshold: 0.90,
                phase2_enabled: true,
                phase1_vocab_ratio: 0.86,
                phase2_min_npmi: 0.10,
                phase2_min_npmi_ascii: 0.10,
                phase2_min_npmi_okuri: 0.10,
                phase2_ascii_quota: 1.0,
                phase2_non_ja_loose: true,
                min_segment_freq: 2,
                phase2_max_segment_chunks: 16,
                phase2_fallback_window: 3,
                beg_strength: 0.0,
                beg_stem_max: 0.7,
                beg_noun_min: 1.8,
                beg_particle_lo: 6.0,
                beg_particle_hi: 9.0,
                beg_phase2: false,
                beg_okuri_boost: 0.0,
                special_tokens: Vec::new(),
            };
        }
        let (mode, byte_mode) = match self {
            Self::StandardBbpe => (PreTokenizerMode::StandardGpt, ByteMode::RawBytes),
            Self::StrictScript => (PreTokenizerMode::StrictScript, ByteMode::Utf8Atomic),
            _ => (PreTokenizerMode::AsymScript, ByteMode::Utf8Atomic),
        };
        TrainConfig {
            name: self.name().to_string(),
            vocab_size,
            pretok: PreTokenizerConfig {
                mode,
                byte_mode,
                ..PreTokenizerConfig::default()
            },
            min_char_freq: 2,
            max_atomic_chars: None,
            seed_unicode_symbols: false,
            min_pair_freq: 2,
            acsg_npmi_beta: match self {
                Self::AsymFull | Self::AsymNoPhase2 | Self::AsymNoPicky => 1.0,
                _ => 0.0,
            },
            picky_enabled: matches!(self, Self::AsymFull | Self::AsymNoPhase2 | Self::AsymNoAcsg),
            picky_ios_threshold: 0.90,
            phase2_enabled: matches!(self, Self::AsymFull | Self::AsymNoPicky | Self::AsymNoAcsg),
            phase1_vocab_ratio: if self == Self::AsymFull { 0.75 } else { 0.85 },
            phase2_min_npmi: if self == Self::AsymFull { 0.45 } else { 0.25 },
            phase2_min_npmi_ascii: if self == Self::AsymFull { 0.10 } else { 0.25 },
            phase2_min_npmi_okuri: if self == Self::AsymFull { 0.45 } else { 0.25 },
            phase2_ascii_quota: if self == Self::AsymFull { 0.30 } else { 1.0 },
            phase2_non_ja_loose: false,
            min_segment_freq: 1,
            phase2_max_segment_chunks: 16,
            phase2_fallback_window: 0,
            beg_strength: if self == Self::AsymFull { 3.0 } else { 0.0 },
            beg_stem_max: 0.7,
            beg_noun_min: 1.8,
            beg_particle_lo: 6.0,
            beg_particle_hi: 9.0,
            beg_phase2: true,
            beg_okuri_boost: 0.0,
            special_tokens: Vec::new(),
        }
    }
}

impl fmt::Display for TrainPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for TrainPreset {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "asym-full" => Ok(Self::AsymFull),
            "asym-no-phase2" => Ok(Self::AsymNoPhase2),
            "asym-no-picky" => Ok(Self::AsymNoPicky),
            "asym-no-acsg" => Ok(Self::AsymNoAcsg),
            "strict-script" => Ok(Self::StrictScript),
            "standard-bbpe" => Ok(Self::StandardBbpe),
            "global" | "asym-global" => Ok(Self::Global),
            other => Err(format!(
                "unknown preset '{other}' (expected one of: {}, global)",
                Self::ALL
                    .iter()
                    .map(|p| p.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_names_roundtrip() {
        for p in TrainPreset::ALL {
            assert_eq!(TrainPreset::from_str(p.name()).unwrap(), p);
            assert_eq!(p.to_string(), p.name());
        }
        assert_eq!(
            TrainPreset::from_str("global").unwrap(),
            TrainPreset::Global
        );
        assert_eq!(
            TrainPreset::from_str("asym-global").unwrap(),
            TrainPreset::Global
        );
        assert_eq!(TrainPreset::ALL.len(), 6);
        assert!(!TrainPreset::ALL.contains(&TrainPreset::Global));
        assert!(TrainPreset::from_str("nope").is_err());
    }

    #[test]
    fn preset_flags_match_spec() {
        let full = TrainPreset::AsymFull.config(4096);
        assert_eq!(full.pretok.mode, PreTokenizerMode::AsymScript);
        assert_eq!(full.pretok.byte_mode, ByteMode::Utf8Atomic);
        assert!(full.acsg_npmi_beta > 0.0 && full.picky_enabled && full.phase2_enabled);

        assert!(!TrainPreset::AsymNoPhase2.config(4096).phase2_enabled);
        assert!(!TrainPreset::AsymNoPicky.config(4096).picky_enabled);
        assert_eq!(TrainPreset::AsymNoAcsg.config(4096).acsg_npmi_beta, 0.0);

        let strict = TrainPreset::StrictScript.config(4096);
        assert_eq!(strict.pretok.mode, PreTokenizerMode::StrictScript);
        assert!(!strict.picky_enabled && !strict.phase2_enabled);

        let bbpe = TrainPreset::StandardBbpe.config(4096);
        assert_eq!(bbpe.pretok.mode, PreTokenizerMode::StandardGpt);
        assert_eq!(bbpe.pretok.byte_mode, ByteMode::RawBytes);

        let global = TrainPreset::Global.config(32_768);
        assert!(global.pretok.multilingual_scripts);
        assert!(global.pretok.attach_space_to_punct);
        assert_eq!(global.max_atomic_chars, Some(18_000));
        assert!(global.seed_unicode_symbols);
        assert_eq!(global.min_segment_freq, 2);
        assert!(global.phase2_non_ja_loose);
        assert_eq!(global.phase1_vocab_ratio, 0.86);
        assert_eq!(global.phase2_fallback_window, 3);
    }
}
