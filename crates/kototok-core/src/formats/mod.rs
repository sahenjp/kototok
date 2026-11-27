//! Import and export for native models and interoperable BPE formats.

mod bpe_files;
mod byte_level;
mod hf;
mod tiktoken;

use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::{
    ByteMode, InternalToken, KTOK_FORMAT_VERSION, KTOK_MAGIC, KototokError, Model, ModelMetadata,
    PreTokenizerConfig, PreTokenizerMode, TokenKind, VocabEvent,
};

/// Serialized model formats available to the generic load and save APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFormat {
    Ktok,
    KototokJson,
    HfJson,
    HfDir,
    Tiktoken,
    BpeFiles,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ktok => "ktok",
            Self::KototokJson => "json",
            Self::HfJson => "hf",
            Self::HfDir => "hf-dir",
            Self::Tiktoken => "tiktoken",
            Self::BpeFiles => "bpe",
        })
    }
}

impl FromStr for ModelFormat {
    type Err = KototokError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "ktok" | "bin" | "binary" => Ok(Self::Ktok),
            "json" | "kototok-json" | "ktok-json" => Ok(Self::KototokJson),
            "hf" | "hf-json" | "huggingface" | "tokenizer.json" => Ok(Self::HfJson),
            "hf-dir" | "huggingface-dir" => Ok(Self::HfDir),
            "tiktoken" => Ok(Self::Tiktoken),
            "bpe" | "bpe-files" | "gpt2" => Ok(Self::BpeFiles),
            _ => Err(invalid(format!("unknown model format {value:?}"))),
        }
    }
}

impl ModelFormat {
    /// Chooses an output format based on the filename or an existing directory.
    pub fn infer_from_save_path(path: &Path) -> Self {
        if path.is_dir() {
            return Self::HfDir;
        }
        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            let name = name.to_ascii_lowercase();
            if name == "tokenizer.json" || name.ends_with(".hf.json") {
                return Self::HfJson;
            }
        }
        if has_extension(path, "tiktoken") {
            Self::Tiktoken
        } else if has_extension(path, "json") {
            Self::KototokJson
        } else {
            Self::Ktok
        }
    }
}

impl Model {
    /// Detects an existing model's format by directory contents or file contents.
    pub fn detect_format(path: impl AsRef<Path>) -> Result<ModelFormat, KototokError> {
        let path = path.as_ref();
        if path.is_dir() {
            detect_directory(path)
        } else {
            detect_file(path, &fs::read(path)?)
        }
    }

    /// Loads a model in an explicitly selected format.
    pub fn load_with_format(
        path: impl AsRef<Path>,
        format: ModelFormat,
    ) -> Result<Self, KototokError> {
        let path = path.as_ref();
        match format {
            ModelFormat::Ktok => Self::load_binary(path),
            ModelFormat::KototokJson => Self::load_json(path),
            ModelFormat::HfJson => Self::load_hf_json(path),
            ModelFormat::HfDir => Self::load_hf_json(path.join("tokenizer.json")),
            ModelFormat::Tiktoken => Self::load_tiktoken(path),
            ModelFormat::BpeFiles => Self::load_bpe_files(path),
        }
    }

    /// Saves a model using the format inferred from the destination.
    pub fn save_auto(&self, path: impl AsRef<Path>) -> Result<(), KototokError> {
        let path = path.as_ref();
        self.save_with_format(path, ModelFormat::infer_from_save_path(path))
    }

    /// Saves a model in an explicitly selected format.
    pub fn save_with_format(
        &self,
        path: impl AsRef<Path>,
        format: ModelFormat,
    ) -> Result<(), KototokError> {
        match format {
            ModelFormat::Ktok => self.save_binary(path),
            ModelFormat::KototokJson => self.save_json(path),
            ModelFormat::HfJson => self.save_hf_json(path),
            ModelFormat::HfDir => self.save_hf_dir(path),
            ModelFormat::Tiktoken => self.save_tiktoken(path),
            ModelFormat::BpeFiles => self.save_bpe_files(path),
        }
    }
}

pub(crate) fn load_auto(path: &Path) -> Result<Model, KototokError> {
    if path.is_dir() {
        return Model::load_with_format(path, detect_directory(path)?);
    }
    let bytes = fs::read(path)?;
    match detect_file(path, &bytes)? {
        ModelFormat::Ktok => Model::from_binary_bytes(&bytes),
        ModelFormat::KototokJson => {
            let model: Model = serde_json::from_slice(&bytes)?;
            model.validate_final_id_sparsity()?;
            Ok(model)
        }
        ModelFormat::HfJson => Model::from_hf_json_bytes(&bytes),
        ModelFormat::Tiktoken => tiktoken::from_bytes(&bytes),
        ModelFormat::HfDir | ModelFormat::BpeFiles => unreachable!("file format detection"),
    }
}

fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
}

fn detect_directory(path: &Path) -> Result<ModelFormat, KototokError> {
    if path.join("tokenizer.json").is_file() {
        Ok(ModelFormat::HfDir)
    } else if path.join("vocab.json").is_file() && path.join("merges.txt").is_file() {
        Ok(ModelFormat::BpeFiles)
    } else {
        Err(invalid(
            "directory has neither tokenizer.json nor a vocab.json/merges.txt pair",
        ))
    }
}

fn detect_file(path: &Path, bytes: &[u8]) -> Result<ModelFormat, KototokError> {
    if bytes.starts_with(&KTOK_MAGIC) {
        return Ok(ModelFormat::Ktok);
    }
    if bytes.trim_ascii_start().starts_with(b"{") {
        let model: Value = serde_json::from_slice(bytes)
            .map_err(|err| invalid(format!("invalid model JSON: {err}")))?;
        if model.get("internal_tokens").is_some() && model.get("events").is_some() {
            return Ok(ModelFormat::KototokJson);
        }
        if model.get("model").is_some() {
            return Ok(ModelFormat::HfJson);
        }
        return Err(invalid("unrecognized model JSON"));
    }
    if has_extension(path, "tiktoken")
        || bytes
            .split(|&byte| byte == b'\n')
            .map(std::str::from_utf8)
            .find(|line| !line.is_ok_and(|line| line.trim().is_empty()))
            .is_some_and(|line| line.is_ok_and(|line| tiktoken::parse_row(line).is_ok()))
    {
        return Ok(ModelFormat::Tiktoken);
    }
    Err(invalid("unrecognized model file"))
}

fn invalid(message: impl Into<String>) -> KototokError {
    KototokError::InvalidFormat(message.into())
}

/// Token bytes paired with the lowest surviving final ID for those bytes.
fn canonical_tokens(model: &Model) -> FxHashMap<&[u8], u32> {
    let mut tokens = FxHashMap::default();
    for token in &model.internal_tokens {
        if let Some(id) = token.final_id {
            tokens
                .entry(token.bytes.as_slice())
                .and_modify(|existing: &mut u32| *existing = (*existing).min(id))
                .or_insert(id);
        }
    }
    tokens
}

struct ImportedModel {
    model: Model,
    next_id: u64,
}

impl ImportedModel {
    fn new(name: &str, mode: PreTokenizerMode, max_id: Option<u32>) -> Self {
        Self {
            model: Model {
                metadata: ModelMetadata {
                    name: name.into(),
                    version: KTOK_FORMAT_VERSION,
                    vocab_size: 0,
                    phase1_vocab_size: 0,
                    phase2_start_rank: 0,
                    pruned_count: 0,
                    invalid_utf8_learned_tokens: 0,
                    train_notes: String::new(),
                },
                pretok: PreTokenizerConfig {
                    mode,
                    byte_mode: ByteMode::RawBytes,
                    ..PreTokenizerConfig::default()
                },
                special_tokens: Vec::new(),
                internal_tokens: (0..=u8::MAX)
                    .map(|byte| InternalToken {
                        internal_id: u32::from(byte),
                        bytes: vec![byte],
                        kind: if byte.is_ascii() {
                            TokenKind::Ascii
                        } else {
                            TokenKind::ByteFallback
                        },
                        final_id: None,
                    })
                    .collect(),
                events: Vec::new(),
                tag_ranges: Vec::new(),
                tagged_prunes: Vec::new(),
            },
            next_id: max_id.map_or(0, |id| u64::from(id) + 1),
        }
    }

    fn allocate_final_id(&mut self) -> Result<u32, KototokError> {
        let id = u32::try_from(self.next_id).map_err(|_| invalid("model token IDs exhausted"))?;
        self.next_id += 1;
        Ok(id)
    }

    fn push_token(
        &mut self,
        bytes: Vec<u8>,
        kind: TokenKind,
        final_id: u32,
    ) -> Result<u32, KototokError> {
        let internal_id = u32::try_from(self.model.internal_tokens.len())
            .map_err(|_| invalid("too many internal tokens"))?;
        self.model.internal_tokens.push(InternalToken {
            internal_id,
            bytes,
            kind,
            final_id: Some(final_id),
        });
        Ok(internal_id)
    }

    fn push_merge(&mut self, left: u32, right: u32, result: u32) {
        self.model.events.push(VocabEvent::Merge {
            left,
            right,
            result,
        });
    }

    fn finish(mut self) -> Result<Model, KototokError> {
        for i in 0..256 {
            if self.model.internal_tokens[i].final_id.is_none() {
                self.model.internal_tokens[i].final_id = Some(self.allocate_final_id()?);
            }
        }
        let model = &mut self.model;
        if model.pretok.mode == PreTokenizerMode::StandardGpt {
            model.metadata.phase2_start_rank =
                u32::try_from(model.events.len()).map_err(|_| invalid("too many merge events"))?;
        }
        model.metadata.vocab_size = model.internal_tokens.len();
        model.metadata.phase1_vocab_size = if model.pretok.mode == PreTokenizerMode::StandardGpt {
            model.metadata.vocab_size
        } else {
            256 + model.special_tokens.len()
        };
        model.metadata.invalid_utf8_learned_tokens =
            model.audit_utf8_validity().invalid_utf8_learned_tokens;

        // The encoder indexes a dense decode table by final ID. Limit empty slots
        // so a small rank file cannot require several GiB just to load its tokenizer.
        self.model.validate_final_id_sparsity()?;
        Ok(self.model)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;
    use crate::Tokenizer;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir()
                .join(format!("kototok-formats-{}-{nonce}", std::process::id()));
            fs::create_dir(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn toy() -> ImportedModel {
        let mut builder =
            ImportedModel::new("test formats", PreTokenizerMode::StandardGpt, Some(255));
        for byte in &mut builder.model.internal_tokens {
            byte.final_id = Some(byte.internal_id);
        }
        builder
    }

    fn add(builder: &mut ImportedModel, text: &str, kind: TokenKind) -> u32 {
        let final_id = builder.allocate_final_id().unwrap();
        builder
            .push_token(text.as_bytes().to_vec(), kind, final_id)
            .unwrap()
    }

    #[test]
    fn format_aliases_and_save_inference() {
        for (format, aliases) in [
            (ModelFormat::Ktok, &["ktok", "bin", "binary"][..]),
            (
                ModelFormat::KototokJson,
                &["json", "kototok-json", "ktok-json"],
            ),
            (
                ModelFormat::HfJson,
                &["hf", "hf-json", "huggingface", "tokenizer.json"],
            ),
            (ModelFormat::HfDir, &["hf-dir", "huggingface-dir"]),
            (ModelFormat::Tiktoken, &["tiktoken"]),
            (ModelFormat::BpeFiles, &["bpe", "bpe-files", "gpt2"]),
        ] {
            assert_eq!(format.to_string(), aliases[0]);
            for alias in aliases {
                assert_eq!(alias.parse::<ModelFormat>().unwrap(), format);
            }
        }
        assert!("unigram".parse::<ModelFormat>().is_err());
        for (name, format) in [
            ("test.KTOK", ModelFormat::Ktok),
            ("no-extension", ModelFormat::Ktok),
            ("test.unknown", ModelFormat::Ktok),
            ("test.JSON", ModelFormat::KototokJson),
            ("Tokenizer.JSON", ModelFormat::HfJson),
            ("test.HF.JSON", ModelFormat::HfJson),
            ("test.TIKTOKEN", ModelFormat::Tiktoken),
        ] {
            assert_eq!(ModelFormat::infer_from_save_path(Path::new(name)), format);
        }
    }

    #[test]
    fn auto_detection_and_roundtrip_for_all_formats() {
        let dir = TempDir::new();
        let mut builder = toy();
        let ab = add(&mut builder, "ab", TokenKind::MergedPhase1);
        let hashtags = add(&mut builder, "##", TokenKind::MergedPhase1);
        builder.push_merge(u32::from(b'a'), u32::from(b'b'), ab);
        builder.push_merge(u32::from(b'#'), u32::from(b'#'), hashtags);
        let eos = add(&mut builder, "<|eos|>", TokenKind::Special);
        let eos_id = builder.model.internal_tokens[eos as usize]
            .final_id
            .unwrap();
        builder
            .model
            .special_tokens
            .push(("<|eos|>".into(), eos_id));
        let model = builder.finish().unwrap();
        let expected = Tokenizer::from_model(model.clone()).unwrap();
        for (name, format) in [
            ("test.ktok", ModelFormat::Ktok),
            ("test.json", ModelFormat::KototokJson),
            ("test.hf.json", ModelFormat::HfJson),
            ("hf", ModelFormat::HfDir),
            ("test.tiktoken", ModelFormat::Tiktoken),
            ("gpt2", ModelFormat::BpeFiles),
        ] {
            let path = dir.0.join(name);
            match format {
                ModelFormat::HfDir => {
                    model.save_with_format(&path, format).unwrap();
                    model.save_auto(&path).unwrap();
                }
                ModelFormat::BpeFiles => model.save_with_format(&path, format).unwrap(),
                _ => model.save_auto(&path).unwrap(),
            }
            assert_eq!(Model::detect_format(&path).unwrap(), format);
            let loaded = Model::load_auto(&path).unwrap();
            assert_eq!(loaded, Model::load_with_format(&path, format).unwrap());
            if matches!(format, ModelFormat::Ktok | ModelFormat::KototokJson) {
                assert_eq!(loaded, model);
            }
            let tokenizer = Tokenizer::from_model(loaded).unwrap();
            assert_eq!(
                tokenizer.encode("ab").unwrap(),
                expected.encode("ab").unwrap()
            );
            assert_eq!(
                tokenizer.encode("##").unwrap(),
                expected.encode("##").unwrap()
            );
            for text in [
                "",
                "abc and Rust",
                "日本語と🦀",
                "fn f() {\n  return 123;\n}",
            ] {
                assert_eq!(
                    tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(),
                    text
                );
            }
            assert_eq!(tokenizer.decode(&[eos_id]).unwrap(), "<|eos|>");
        }

        let hf = dir.0.join("hf");
        let config = fs::read_to_string(hf.join("tokenizer_config.json")).unwrap();
        assert!(config.contains("1000000000000000019884624838656,"));
        let config: Value = serde_json::from_str(&config).unwrap();
        assert_eq!(config["tokenizer_class"], "PreTrainedTokenizerFast");
        assert_eq!(config["clean_up_tokenization_spaces"], false);
        assert_eq!(
            config["added_tokens_decoder"][eos_id.to_string()]["content"],
            "<|eos|>"
        );
        assert!(
            config["added_tokens_decoder"][eos_id.to_string()]
                .get("id")
                .is_none()
        );
        let map: Value =
            serde_json::from_slice(&fs::read(hf.join("special_tokens_map.json")).unwrap()).unwrap();
        assert_eq!(map, json!({"eos_token": "<|eos|>"}));

        let bpe = dir.0.join("gpt2");
        let merges = bpe.join("merges.txt");
        let text = fs::read_to_string(&merges).unwrap();
        assert!(text.starts_with("#version: 0.2\n"));
        fs::write(&merges, format!("# custom header\n#comment\n\n{text}")).unwrap();
        let pair = Model::load_bpe_pair(bpe.join("vocab.json"), merges).unwrap();
        assert_eq!(
            Tokenizer::from_model(pair).unwrap().encode("##").unwrap(),
            expected.encode("##").unwrap()
        );

        let path = dir.0.join("renamed");
        model.save_tiktoken(&path).unwrap();
        assert_eq!(Model::detect_format(path).unwrap(), ModelFormat::Tiktoken);
    }

    #[test]
    fn auto_detection_rejects_corrupt_models() {
        let dir = TempDir::new();
        assert!(matches!(
            Model::detect_format(&dir.0),
            Err(KototokError::InvalidFormat(_))
        ));
        for (name, data, detection) in [
            ("broken-json", b" {".as_slice(), None),
            ("alien", b"{} ", None),
            ("unknown", b"???", None),
            ("native", b"KTOK", Some(ModelFormat::Ktok)),
            (
                "native-json",
                b"\n { \"internal_tokens\":null, \"events\": [] }",
                Some(ModelFormat::KototokJson),
            ),
            ("hf-json", b"\t{\"model\":null}", Some(ModelFormat::HfJson)),
            ("broken.tiktoken", b"bad rank", Some(ModelFormat::Tiktoken)),
            (
                "broken-ranks",
                b"\nYQ== 0\n\xff",
                Some(ModelFormat::Tiktoken),
            ),
        ] {
            let path = dir.0.join(name);
            fs::write(&path, data).unwrap();
            assert_eq!(Model::detect_format(&path).ok(), detection, "{name}");
            assert!(Model::load_auto(&path).is_err(), "{name}");
        }
    }

    #[test]
    fn hf_export_keeps_atomic_and_picky_merges_but_omits_bko_and_unusable_merges() {
        let mut builder = toy();
        builder.model.pretok = PreTokenizerConfig::default();
        let first = add(&mut builder, "日", TokenKind::AtomicChar);
        let second = add(&mut builder, "本", TokenKind::AtomicChar);
        let third = add(&mut builder, "の", TokenKind::AtomicChar);
        let japan = add(&mut builder, "日本", TokenKind::MergedPhase1);
        builder.push_merge(first, second, japan);

        let bad = builder
            .push_token("日本の".as_bytes().to_vec(), TokenKind::MergedPhase1, 0)
            .unwrap();
        builder.model.internal_tokens[bad as usize].final_id = None;
        builder.push_merge(japan, third, bad);
        builder.model.events.push(VocabEvent::Prune {
            target: bad,
            split_into: vec![japan, third],
        });

        let ab = builder
            .push_token(b"ab".to_vec(), TokenKind::MergedPhase1, 0)
            .unwrap();
        builder.model.internal_tokens[ab as usize].final_id = None;
        builder.push_merge(97, 98, ab);
        let abc = add(&mut builder, "abc", TokenKind::MergedPhase1);
        builder.push_merge(ab, 99, abc);
        builder.model.events.push(VocabEvent::Prune {
            target: ab,
            split_into: vec![97, 98],
        });
        let duplicate = add(&mut builder, "abc", TokenKind::MergedPhase1);
        let hole = builder.model.internal_tokens[duplicate as usize]
            .final_id
            .unwrap();
        let special = add(&mut builder, "<|資料|>", TokenKind::Special);
        let special_id = builder.model.internal_tokens[special as usize]
            .final_id
            .unwrap();
        builder
            .model
            .special_tokens
            .push(("<|資料|>".into(), special_id));
        let impossible = add(&mut builder, "abd", TokenKind::MergedPhase2);
        builder.push_merge(ab, 100, impossible);

        let model = builder.finish().unwrap();
        let json = model.to_hf_json_string().unwrap();
        let mut exported: Value = serde_json::from_str(&json).unwrap();
        let vocab = exported["model"]["vocab"].as_object().unwrap();
        let mut ids: Vec<_> = vocab.values().map(|id| id.as_u64().unwrap()).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..vocab.len() as u64).collect::<Vec<_>>());
        assert_eq!(vocab["abc"], 260);
        assert_eq!(vocab["æĹ"], hole);
        assert_eq!(vocab["<|資料|>"], special_id);
        assert_eq!(exported["added_tokens"][0]["id"], special_id);
        assert_eq!(exported["added_tokens"][0]["special"], true);
        assert_eq!(exported["pre_tokenizer"]["type"], "Sequence");
        assert_eq!(
            exported["pre_tokenizer"]["pretokenizers"][4]["use_regex"],
            false
        );
        let merges = exported["model"]["merges"].as_array().unwrap();
        assert_eq!(merges[0], "æ Ĺ");
        assert_eq!(merges[1], "æĹ ¥");
        assert!(merges.contains(&json!("a b")));
        assert!(merges.contains(&json!("ab c")));
        assert!(!merges.contains(&json!("ab d")));
        assert!(merges.iter().all(|merge| {
            let (left, right) = merge.as_str().unwrap().split_once(' ').unwrap();
            vocab.contains_key(left)
                && vocab.contains_key(right)
                && vocab.contains_key(&format!("{left}{right}"))
        }));
        let tok =
            Tokenizer::from_model(Model::from_hf_json_bytes(json.as_bytes()).unwrap()).unwrap();
        let original = Tokenizer::from_model(model).unwrap();
        for text in ["日本", "日本の", "abc", "a bc", "🦀"] {
            assert_eq!(
                tok.encode(text).unwrap(),
                original.encode(text).unwrap(),
                "{text}"
            );
        }
        assert_eq!(tok.encode("日本の").unwrap(), [259, 258]);
        assert_eq!(tok.decode(&[special_id]).unwrap(), "<|資料|>");

        let pairs: Vec<_> = merges
            .iter()
            .map(|merge| {
                let (left, right) = merge.as_str().unwrap().split_once(' ').unwrap();
                [left.to_owned(), right.to_owned()]
            })
            .collect();
        exported["model"]["merges"] = json!(pairs);
        let pair_model =
            Model::from_hf_json_bytes(&serde_json::to_vec(&exported).unwrap()).unwrap();
        assert_eq!(&pair_model, tok.model());
    }

    #[test]
    fn hf_export_compacts_remaining_holes_and_updates_special_ids() {
        let mut builder = toy();
        builder
            .push_token(b"<pad>".to_vec(), TokenKind::Special, 900)
            .unwrap();
        builder.model.special_tokens.push(("<pad>".into(), 900));
        let model = builder.finish().unwrap();
        let value: Value = serde_json::from_str(&model.to_hf_json_string().unwrap()).unwrap();
        assert_eq!(value["pre_tokenizer"]["use_regex"], true);
        assert_eq!(value["model"]["vocab"]["<pad>"], 256);
        assert_eq!(value["added_tokens"][0]["id"], 256);
    }

    #[test]
    fn hf_import_supports_raw_unicode_metaspace_and_byte_fallback() {
        let value = json!({
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "▁": 1, "▁a": 2, "日": 3, "<0xE8>": 4, "<0xA9>": 5, "<0xB1>": 6, "<0x61>": 7},
                "merges": [["▁", "a"]],
            },
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
            ]},
            "added_tokens": [{"id": 8, "content": "<special>", "special": true}],
        });
        let model = Model::from_hf_json_bytes(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(model.pretok.byte_mode, ByteMode::RawBytes);
        assert_eq!(model.pretok.mode, PreTokenizerMode::StandardGpt);
        assert_eq!(
            model.metadata.phase2_start_rank as usize,
            model.events.len()
        );
        assert_eq!(model.special_tokens, [("<special>".into(), 8)]);
        let tok = Tokenizer::from_model(model).unwrap();
        assert_eq!(tok.encode(" a日").unwrap(), [2, 3]);
        assert_eq!(
            tok.decode(&[2, 3, 4, 5, 6, 7, 8]).unwrap(),
            " a日話a<special>"
        );
        let arbitrary = "日本🦀abc";
        assert_eq!(
            tok.decode(&tok.encode(arbitrary).unwrap()).unwrap(),
            arbitrary
        );

        let raw = json!({
            "model": {"type": "BPE", "vocab": {"日": 0, "本": 1, "日本": 2}, "merges": ["日 本"]},
        });
        let raw = Model::from_hf_json_bytes(&serde_json::to_vec(&raw).unwrap()).unwrap();
        assert_eq!(
            Tokenizer::from_model(raw).unwrap().encode("日本").unwrap(),
            [2]
        );
    }

    #[test]
    fn hf_import_rejects_invalid_shapes_ids_and_merge_operands() {
        let unsupported = br#"{"model":{"type":"Unigram","vocab":[]}}"#;
        assert!(
            matches!(Model::from_hf_json_bytes(unsupported), Err(KototokError::InvalidFormat(msg)) if msg.contains("only Hugging Face BPE"))
        );
        for invalid_json in [
            r#"{}"#,
            r#"{"model":{"type":"BPE"}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":-1},"merges":[]}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":4294967296},"merges":[]}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":4294967295},"merges":[]}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":0,"b":0},"merges":[]}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[["a"]]}}"#,
            r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]}}"#,
        ] {
            assert!(
                matches!(
                    Model::from_hf_json_bytes(invalid_json.as_bytes()),
                    Err(KototokError::InvalidFormat(_))
                ),
                "{invalid_json}"
            );
        }
    }

    #[test]
    fn hf_import_plain_byte_level_without_kototok_split_stays_standard_gpt() {
        let plain = json!({
            "pre_tokenizer": {"type": "ByteLevel", "use_regex": false},
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
        });
        let model = Model::from_hf_json_bytes(&serde_json::to_vec(&plain).unwrap()).unwrap();
        assert_eq!(model.pretok.mode, PreTokenizerMode::StandardGpt);

        let kototok_split = json!({
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"[\u3000-\u3004]"}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "use_regex": false},
            ]},
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]},
        });
        let model =
            Model::from_hf_json_bytes(&serde_json::to_vec(&kototok_split).unwrap()).unwrap();
        assert_eq!(model.pretok.mode, PreTokenizerMode::AsymScript);
    }

    #[test]
    fn hf_export_rejects_special_collision_with_distinct_ordinary_id() {
        let mut builder = toy();
        let ab = add(&mut builder, "ab", TokenKind::MergedPhase1);
        let ab_id = builder.model.internal_tokens[ab as usize].final_id.unwrap();
        builder.model.special_tokens.push(("ab".into(), ab_id));
        let model = builder.finish().unwrap();
        assert!(model.to_hf_json_string().is_ok());

        let mut builder = toy();
        add(&mut builder, "ab", TokenKind::MergedPhase1);
        builder.model.special_tokens.push(("ab".into(), 999));
        let model = builder.finish().unwrap();
        assert!(matches!(
            model.to_hf_json_string(),
            Err(KototokError::InvalidFormat(_))
        ));
    }

    #[test]
    fn hf_export_global_regex_uses_soft_ascii_punct_and_spaced_letters() {
        let mut builder = toy();
        builder.model.pretok.mode = PreTokenizerMode::AsymScript;
        builder.model.pretok.multilingual_scripts = true;
        builder.model.pretok.attach_space_to_punct = true;
        let model = builder.finish().unwrap();
        let value: Value = serde_json::from_str(&model.to_hf_json_string().unwrap()).unwrap();
        let pretokenizers = value["pre_tokenizer"]["pretokenizers"].as_array().unwrap();
        let space_regex = pretokenizers[2]["pattern"]["Regex"].as_str().unwrap();
        let punct_regex = pretokenizers[3]["pattern"]["Regex"].as_str().unwrap();
        assert!(space_regex.contains(r"\p{Han}"));
        assert!(space_regex.contains('('));
        assert_eq!(punct_regex, r#"(?<=[^\x00-\x7F])(?=[()\[\]{},.;:!?"'`])"#);

        let reimported =
            Model::from_hf_json_bytes(model.to_hf_json_string().unwrap().as_bytes()).unwrap();
        assert_eq!(reimported.pretok.mode, PreTokenizerMode::AsymScript);
        assert!(reimported.pretok.multilingual_scripts);
        assert!(reimported.pretok.attach_space_to_punct);
    }

    #[test]
    fn hf_roundtrip_preserves_strict_script_mode() {
        let mut builder = toy();
        builder.model.pretok.mode = PreTokenizerMode::StrictScript;
        let model = builder.finish().unwrap();
        let json = model.to_hf_json_string().unwrap();
        let reimported = Model::from_hf_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(reimported.pretok.mode, PreTokenizerMode::StrictScript);
        assert!(!reimported.pretok.multilingual_scripts);
        assert!(!reimported.pretok.attach_space_to_punct);
    }
}
