//! `kototok-train`: parallel trainer for the AsymScript-BPE (`kototok`)
//! tokenizer. See `docs/ALGORITHM.md` for the formal specification.

mod config;
mod counter;
mod trainer;
mod vocab_init;

pub use config::{TrainConfig, TrainPreset};
pub use counter::CorpusCounts;
pub use trainer::{TrainError, train};

pub use kototok_core::{ByteMode, Model, PreTokenizerConfig, PreTokenizerMode};

#[cfg(test)]
mod tests;
