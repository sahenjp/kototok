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
        }
    }

    /// Builds the `TrainConfig` for this preset at `vocab_size`.
    pub fn config(self, vocab_size: usize) -> TrainConfig {
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
            min_pair_freq: 2,
            acsg_npmi_beta: match self {
                Self::AsymFull | Self::AsymNoPhase2 | Self::AsymNoPicky => 1.0,
                _ => 0.0,
            },
            picky_enabled: matches!(
                self,
                Self::AsymFull | Self::AsymNoPhase2 | Self::AsymNoAcsg
            ),
            picky_ios_threshold: 0.90,
            phase2_enabled: matches!(
                self,
                Self::AsymFull | Self::AsymNoPicky | Self::AsymNoAcsg
            ),
            phase1_vocab_ratio: 0.85,
            phase2_min_npmi: 0.25,
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
            other => Err(format!(
                "unknown preset '{other}' (expected one of: {})",
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
    }
}
