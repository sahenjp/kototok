//! The GPT-2-style vocab.json and merges.txt pair.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::hf::{Merge, Vocab, build_export, import_bpe_files, write_json};
use super::invalid;
use crate::{KototokError, Model};

impl Model {
    /// Loads vocab.json and merges.txt from a GPT-2-style BPE directory.
    pub fn load_bpe_files(dir: impl AsRef<Path>) -> Result<Self, KototokError> {
        let dir = dir.as_ref();
        Self::load_bpe_pair(dir.join("vocab.json"), dir.join("merges.txt"))
    }

    /// Loads a BPE vocabulary and merge list from two explicit file paths.
    pub fn load_bpe_pair(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
    ) -> Result<Self, KototokError> {
        let vocab: Vocab = serde_json::from_slice(&fs::read(vocab_path)?)
            .map_err(|err| invalid(format!("invalid BPE vocab.json: {err}")))?;
        let merges = fs::read_to_string(merges_path)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with("#version:"))
            .filter_map(|line| {
                let merge = Merge::Text(line.to_owned());
                // A literal '#' is a valid ByteLevel merge operand.
                if line.starts_with('#')
                    && !merge.operands().is_ok_and(|(left, right)| {
                        vocab.contains_key(left)
                            && vocab.contains_key(right)
                            && vocab.contains_key(&format!("{left}{right}"))
                    })
                {
                    None
                } else {
                    Some(merge)
                }
            })
            .collect();
        import_bpe_files(vocab, merges)
    }

    /// Writes a ByteLevel BPE vocabulary and ordered merge list to a directory.
    pub fn save_bpe_files(&self, dir: impl AsRef<Path>) -> Result<(), KototokError> {
        let dir = dir.as_ref();
        let bpe = build_export(self, false)?;
        fs::create_dir_all(dir)?;
        write_json(&dir.join("vocab.json"), &bpe.vocab)?;
        let mut writer = BufWriter::new(fs::File::create(dir.join("merges.txt"))?);
        writeln!(writer, "#version: 0.2")?;
        for merge in bpe.merges {
            writeln!(writer, "{merge}")?;
        }
        writer.flush()?;
        Ok(())
    }
}
