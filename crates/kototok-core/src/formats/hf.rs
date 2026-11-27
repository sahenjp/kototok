//! The Hugging Face BPE model and its directory sidecars.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::byte_level::{bytes_to_unicode, decode_byte_level, encode_byte_level};
use super::{ImportedModel, canonical_tokens, invalid};
use crate::{
    ByteMode, KototokError, Model, PreTokenizerConfig, PreTokenizerMode, TokenKind, VocabEvent,
};

pub(super) type Vocab = BTreeMap<String, u32>;

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum Merge {
    Text(String),
    Pair([String; 2]),
}

impl Merge {
    pub(super) fn operands(&self) -> Result<(&str, &str), KototokError> {
        match self {
            Self::Pair([left, right]) if !left.is_empty() && !right.is_empty() => Ok((left, right)),
            Self::Text(line) => line
                .split_once(' ')
                .filter(|(left, right)| {
                    !left.is_empty() && !right.is_empty() && !right.contains(' ')
                })
                .ok_or_else(|| invalid(format!("invalid BPE merge {line:?}"))),
            Self::Pair(_) => Err(invalid("empty BPE merge operand")),
        }
    }
}

#[derive(Deserialize)]
struct HfImport {
    model: BpeImport,
    #[serde(default)]
    added_tokens: Vec<AddedTokenImport>,
    #[serde(default)]
    decoder: Value,
    #[serde(default)]
    pre_tokenizer: Value,
}

#[derive(Deserialize)]
struct BpeImport {
    vocab: Vocab,
    merges: Vec<Merge>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
}

#[derive(Deserialize)]
struct AddedTokenImport {
    id: u32,
    content: String,
}

#[derive(Serialize)]
struct AddedToken {
    id: u32,
    #[serde(flatten)]
    properties: AddedTokenProperties,
}

#[derive(Serialize)]
struct AddedTokenProperties {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

pub(super) struct BpeExport {
    pub(super) vocab: Vocab,
    pub(super) merges: Vec<String>,
    added_tokens: Vec<AddedToken>,
}

#[derive(Serialize)]
struct HfExport<'a> {
    version: &'static str,
    truncation: Option<()>,
    padding: Option<()>,
    added_tokens: &'a [AddedToken],
    normalizer: Option<()>,
    pre_tokenizer: Value,
    post_processor: Option<()>,
    decoder: Value,
    model: BpeModel<'a>,
}

#[derive(Serialize)]
struct BpeModel<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    dropout: Option<f32>,
    unk_token: Option<&'a str>,
    continuing_subword_prefix: Option<()>,
    end_of_word_suffix: Option<()>,
    fuse_unk: bool,
    byte_fallback: bool,
    vocab: &'a Vocab,
    merges: &'a [String],
}

impl Model {
    /// Loads a Hugging Face BPE tokenizer.json.
    pub fn load_hf_json(path: impl AsRef<Path>) -> Result<Self, KototokError> {
        Self::from_hf_json_bytes(&fs::read(path)?)
    }

    /// Parses the BPE vocabulary, merge ranks, and added tokens from tokenizer.json.
    pub fn from_hf_json_bytes(bytes: &[u8]) -> Result<Self, KototokError> {
        // Inspect the type first so non-BPE models receive a useful error even
        // when their vocab and merge fields have a different shape.
        #[derive(Deserialize)]
        struct Header {
            model: ModelType,
        }
        #[derive(Deserialize)]
        struct ModelType {
            #[serde(rename = "type")]
            kind: String,
        }

        let bad_json = |err| invalid(format!("invalid Hugging Face tokenizer.json: {err}"));
        let header: Header = serde_json::from_slice(bytes).map_err(bad_json)?;
        if header.model.kind != "BPE" {
            return Err(invalid(format!(
                "only Hugging Face BPE models are supported (found {:?})",
                header.model.kind
            )));
        }

        let HfImport {
            model,
            added_tokens,
            decoder,
            pre_tokenizer,
        } = serde_json::from_slice(bytes).map_err(bad_json)?;
        let BpeImport {
            vocab,
            merges,
            byte_fallback,
            continuing_subword_prefix,
            end_of_word_suffix,
        } = model;
        if [continuing_subword_prefix, end_of_word_suffix]
            .iter()
            .flatten()
            .any(|affix| !affix.is_empty())
        {
            return Err(invalid(
                "Hugging Face BPE subword prefixes and suffixes are unsupported",
            ));
        }

        let byte_level = find_component(&pre_tokenizer, "ByteLevel");
        let encoding = if byte_level.is_some() || find_component(&decoder, "ByteLevel").is_some() {
            Encoding::ByteLevel
        } else if has_metaspace(&decoder) || has_metaspace(&pre_tokenizer) {
            Encoding::Metaspace
        } else if (0..=u8::MAX)
            .all(|b| vocab.contains_key(bytes_to_unicode(b).encode_utf8(&mut [0; 4])))
        {
            Encoding::ByteLevel
        } else {
            Encoding::Raw
        };
        let is_kototok_script = byte_level
            .and_then(|value| value.get("use_regex"))
            .and_then(Value::as_bool)
            == Some(false)
            && has_kototok_split(&pre_tokenizer);
        let is_strict = is_kototok_script
            && has_split_matching(&pre_tokenizer, |p| p == STRICT_KANJI_HIRA_REGEX);
        let mode = if is_strict {
            PreTokenizerMode::StrictScript
        } else if is_kototok_script {
            PreTokenizerMode::AsymScript
        } else {
            PreTokenizerMode::StandardGpt
        };
        let multilingual_scripts = is_kototok_script
            && has_split_matching(&pre_tokenizer, |p| p.contains(r"[\p{L}&&[^\p{Han}"));
        let attach_space_to_punct = is_kototok_script
            && has_split_matching(&pre_tokenizer, |p| p == hard_punct_regex(true));
        let byte_fallback = byte_fallback || find_component(&decoder, "ByteFallback").is_some();
        let mut imported = import_bpe(
            "Hugging Face BPE",
            vocab,
            merges,
            added_tokens,
            encoding,
            byte_fallback,
            mode,
        )?;
        imported.pretok.multilingual_scripts = multilingual_scripts;
        imported.pretok.attach_space_to_punct = attach_space_to_punct;
        Ok(imported)
    }

    /// Serializes the model to a Hugging Face BPE tokenizer.json string.
    pub fn to_hf_json_string(&self) -> Result<String, KototokError> {
        let bpe = build_export(self, true)?;
        Ok(serde_json::to_string_pretty(&self.hf_export(&bpe))?)
    }

    /// Writes a Hugging Face BPE tokenizer.json.
    pub fn save_hf_json(&self, path: impl AsRef<Path>) -> Result<(), KototokError> {
        let bpe = build_export(self, true)?;
        write_json(path.as_ref(), &self.hf_export(&bpe))
    }

    /// Writes tokenizer.json and the two Hugging Face tokenizer configuration files.
    pub fn save_hf_dir(&self, dir: impl AsRef<Path>) -> Result<(), KototokError> {
        let dir = dir.as_ref();
        let bpe = build_export(self, true)?;
        let tokenizer = self.hf_export(&bpe);
        let special_tokens = named_special_tokens(&bpe.added_tokens);
        let decoder: BTreeMap<u32, &AddedTokenProperties> = bpe
            .added_tokens
            .iter()
            .map(|token| (token.id, &token.properties))
            .collect();
        let mut config = json!({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": false,
            "added_tokens_decoder": decoder,
        });
        for (&key, &token) in &special_tokens {
            config[key] = json!(token);
        }
        // Preserve the exact integer used by Transformers beyond serde_json's u64 range.
        let config = serde_json::to_string_pretty(&config)?;
        fs::create_dir_all(dir)?;
        write_json(&dir.join("tokenizer.json"), &tokenizer)?;
        fs::write(
            dir.join("tokenizer_config.json"),
            format!(
                "{{\n  \"model_max_length\": 1000000000000000019884624838656,{}",
                &config[1..]
            ),
        )?;
        write_json(&dir.join("special_tokens_map.json"), &special_tokens)
    }

    fn hf_export<'a>(&self, bpe: &'a BpeExport) -> HfExport<'a> {
        HfExport {
            version: "1.0",
            truncation: None,
            padding: None,
            added_tokens: &bpe.added_tokens,
            normalizer: None,
            pre_tokenizer: pre_tokenizer(self.pretok.mode, &self.pretok),
            post_processor: None,
            decoder: byte_level(false),
            model: BpeModel {
                kind: "BPE",
                dropout: None,
                unk_token: named_special_tokens(&bpe.added_tokens)
                    .get("unk_token")
                    .copied(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                fuse_unk: false,
                byte_fallback: false,
                vocab: &bpe.vocab,
                merges: &bpe.merges,
            },
        }
    }
}

pub(super) fn write_json(path: &Path, value: &impl Serialize) -> Result<(), KototokError> {
    let mut writer = BufWriter::new(fs::File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.flush()?;
    Ok(())
}

fn named_special_tokens(tokens: &[AddedToken]) -> BTreeMap<&'static str, &str> {
    let mut named = BTreeMap::new();
    for token in tokens {
        let content = &token.properties.content;
        let role = match content.to_ascii_lowercase().as_str() {
            "<s>" | "<bos>" | "<|bos|>" | "<|begin_of_text|>" | "[bos]" => "bos_token",
            "</s>" | "<eos>" | "<|eos|>" | "<|end_of_text|>" | "<|endoftext|>" | "<|eot_id|>"
            | "[eos]" => "eos_token",
            "<unk>" | "<|unk|>" | "[unk]" => "unk_token",
            "<pad>" | "<|pad|>" | "<|padding|>" | "[pad]" => "pad_token",
            _ => continue,
        };
        named.entry(role).or_insert(content.as_str());
    }
    named
}

fn byte_level(use_regex: bool) -> Value {
    json!({
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": use_regex,
    })
}

fn hard_punct_regex(attach_space_to_punct: bool) -> &'static str {
    if attach_space_to_punct {
        r#"(?<=[^\x00-\x7F])(?=[()\[\]{},.;:!?"'`])"#
    } else {
        r#"(?<=[A-Za-z0-9_]|[^\x00-\x7F])(?=[()\[\]{},.;:!?"'`])"#
    }
}

fn space_split_regex(multilingual: bool, attach_space_to_punct: bool) -> String {
    let mut ascii = String::from(r"A-Za-z0-9_+\-*/=<>!&|^~@#$%\\");
    if attach_space_to_punct {
        ascii.push_str(r"()\[\]{},.;:!?");
        ascii.push('"');
        ascii.push_str("'`");
    }
    let mut regex = format!(r"[ \t]{{2,}}|\t+| (?![{ascii}]");
    if multilingual {
        // Every non-CJK alphabetic script attaches a single leading space.
        regex.push_str(r"|[\p{L}&&[^\p{Han}\p{Hiragana}\p{Katakana}]]");
    }
    regex.push(')');
    regex
}

const STRICT_KANJI_HIRA_REGEX: &str =
    r"(?<=[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u3005-\u3007])(?=[\u3041-\u3096\u309D-\u309F])";

fn pre_tokenizer(mode: PreTokenizerMode, cfg: &PreTokenizerConfig) -> Value {
    if mode == PreTokenizerMode::StandardGpt {
        return byte_level(true);
    }
    let split = |regex: &str, behavior: &str| {
        json!({
            "type": "Split",
            "pattern": { "Regex": regex },
            "behavior": behavior,
            "invert": false,
        })
    };
    let ja_punct = if cfg.attach_space_to_punct {
        r"[\u3000-\u3004\u3008-\u301F\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65]+|(?<![\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FD-\u30FF\uFF66-\uFF9F\u30FC])\u30FC"
    } else {
        r"[\u3000-\u3004\u3008-\u301F\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65]|(?<![\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FD-\u30FF\uFF66-\uFF9F\u30FC])\u30FC"
    };
    let spaces = space_split_regex(cfg.multilingual_scripts, cfg.attach_space_to_punct);
    let mut pretokenizers = vec![
        split(ja_punct, "Isolated"),
        split(r"\r\n|\r|\n(?![ \t\n])", "MergedWithPrevious"),
        split(&spaces, "MergedWithPrevious"),
        split(hard_punct_regex(cfg.attach_space_to_punct), "Isolated"),
    ];
    if mode == PreTokenizerMode::StrictScript {
        pretokenizers.push(split(STRICT_KANJI_HIRA_REGEX, "Isolated"));
    }
    pretokenizers.push(byte_level(false));
    json!({
        "type": "Sequence",
        "pretokenizers": pretokenizers,
    })
}

fn split_pattern(value: &Value) -> Option<&str> {
    if value.get("type").and_then(Value::as_str) != Some("Split") {
        return None;
    }
    value
        .pointer("/pattern/Regex")
        .or_else(|| value.pointer("/pattern/String"))
        .and_then(Value::as_str)
}

fn has_split_matching(value: &Value, predicate: impl Fn(&str) -> bool + Copy) -> bool {
    if split_pattern(value).is_some_and(predicate) {
        return true;
    }
    value
        .get("pretokenizers")
        .and_then(Value::as_array)
        .is_some_and(|children| {
            children
                .iter()
                .any(|child| has_split_matching(child, predicate))
        })
}

fn has_kototok_split(value: &Value) -> bool {
    has_split_matching(value, |pattern| {
        pattern.contains(r"\u3000-\u3004")
            || pattern.contains(r"\u{3000}-\u{3004}")
            || pattern.contains('\u{3000}')
    })
}

fn find_component<'a>(value: &'a Value, kind: &str) -> Option<&'a Value> {
    if value.get("type").and_then(Value::as_str) == Some(kind) {
        return Some(value);
    }
    ["decoders", "pretokenizers"]
        .into_iter()
        .filter_map(|key| value.get(key).and_then(Value::as_array))
        .flatten()
        .find_map(|child| find_component(child, kind))
}

fn has_metaspace(value: &Value) -> bool {
    match value.get("type").and_then(Value::as_str) {
        Some("Metaspace") => true,
        Some("Replace") => {
            value.pointer("/pattern/String").and_then(Value::as_str) == Some("▁")
                && value.get("content").and_then(Value::as_str) == Some(" ")
        }
        Some("Sequence") => ["decoders", "pretokenizers"]
            .into_iter()
            .filter_map(|key| value.get(key).and_then(Value::as_array))
            .flatten()
            .any(has_metaspace),
        _ => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Encoding {
    ByteLevel,
    Metaspace,
    Raw,
}

fn fallback_byte(symbol: &str) -> Option<u8> {
    let hex = symbol.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return u8::from_str_radix(hex, 16).ok();
    }
    None
}

pub(super) fn import_bpe_files(vocab: Vocab, merges: Vec<Merge>) -> Result<Model, KototokError> {
    import_bpe(
        "BPE files",
        vocab,
        merges,
        Vec::new(),
        Encoding::ByteLevel,
        false,
        PreTokenizerMode::StandardGpt,
    )
}

fn import_bpe(
    name: &str,
    vocab: Vocab,
    merges: Vec<Merge>,
    added_tokens: Vec<AddedTokenImport>,
    encoding: Encoding,
    byte_fallback: bool,
    mode: PreTokenizerMode,
) -> Result<Model, KototokError> {
    if vocab.is_empty() {
        return Err(invalid("empty BPE vocabulary"));
    }

    let mut added_by_id = FxHashMap::default();
    let mut added_names = FxHashSet::default();
    for token in &added_tokens {
        if token.content.is_empty()
            || !added_names.insert(token.content.as_str())
            || added_by_id
                .insert(token.id, token.content.as_str())
                .is_some()
        {
            return Err(invalid("empty or duplicate Hugging Face added token"));
        }
    }

    let max_id = vocab
        .values()
        .copied()
        .chain(added_by_id.keys().copied())
        .max();
    let mut builder = ImportedModel::new(name, mode, max_id);
    let mut ids_by_symbol = FxHashMap::default();
    let mut used_ids = FxHashMap::default();
    let mut scalar_ids = Vec::new();
    let mut entries: Vec<_> = vocab.iter().collect();
    // If both 'a' and '<0x61>' exist, prefer the ordinary character for the byte seed.
    entries.sort_unstable_by_key(|&(symbol, &id)| {
        (
            byte_fallback && fallback_byte(symbol).is_some(),
            encoding == Encoding::Metaspace && symbol == " ",
            id,
        )
    });
    for (symbol, &id) in entries {
        if symbol.is_empty() || used_ids.contains_key(&id) {
            return Err(invalid(format!(
                "empty BPE token or duplicate vocabulary ID {id}"
            )));
        }
        let literal_added = added_by_id
            .get(&id)
            .is_some_and(|content| *content == symbol);
        let fallback = byte_fallback.then(|| fallback_byte(symbol)).flatten();
        let bytes = if literal_added {
            symbol.as_bytes().to_vec()
        } else if let Some(byte) = fallback {
            vec![byte]
        } else {
            match encoding {
                Encoding::ByteLevel => decode_byte_level(symbol).ok_or_else(|| {
                    invalid(format!(
                        "token {symbol:?} is outside the ByteLevel alphabet"
                    ))
                })?,
                Encoding::Metaspace => symbol.replace('▁', " ").into_bytes(),
                Encoding::Raw => symbol.as_bytes().to_vec(),
            }
        };
        let is_scalar = encoding != Encoding::ByteLevel
            && !literal_added
            && bytes.len() > 1
            && symbol.chars().count() == 1;
        let iid = match bytes.as_slice() {
            &[byte]
                if builder.model.internal_tokens[usize::from(byte)]
                    .final_id
                    .is_none() =>
            {
                builder.model.internal_tokens[usize::from(byte)].final_id = Some(id);
                u32::from(byte)
            }
            _ => builder.push_token(
                bytes,
                if literal_added {
                    TokenKind::Special
                } else {
                    TokenKind::MergedPhase1
                },
                id,
            )?,
        };
        if is_scalar {
            scalar_ids.push(iid);
        }
        ids_by_symbol.insert(symbol.as_str(), iid);
        used_ids.insert(id, iid);
    }

    for token in added_tokens {
        if let Some(&iid) = used_ids.get(&token.id) {
            if builder.model.internal_tokens[iid as usize].bytes != token.content.as_bytes() {
                return Err(invalid(format!(
                    "added token {:?} conflicts with vocabulary ID {}",
                    token.content, token.id
                )));
            }
        } else {
            builder.push_token(
                token.content.as_bytes().to_vec(),
                TokenKind::Special,
                token.id,
            )?;
        }
        builder.model.special_tokens.push((token.content, token.id));
    }

    // Raw/Metaspace BPE starts with Unicode scalars. Assemble those from the
    // imported model's byte seeds before replaying the original merge ranks.
    let mut prefixes = FxHashMap::default();
    for iid in scalar_ids {
        let bytes = builder.model.internal_tokens[iid as usize].bytes.clone();
        let mut left = u32::from(bytes[0]);
        for end in 2..=bytes.len() {
            let prefix = &bytes[..end];
            let right = u32::from(bytes[end - 1]);
            left = if let Some(&existing) = prefixes.get(prefix) {
                existing
            } else {
                let result = if end == bytes.len() {
                    iid
                } else {
                    let final_id = builder.allocate_final_id()?;
                    builder.push_token(prefix.to_vec(), TokenKind::MergedPhase1, final_id)?
                };
                builder.push_merge(left, right, result);
                prefixes.insert(prefix.to_vec(), result);
                result
            };
        }
    }

    for merge in merges {
        let (left, right) = merge.operands()?;
        let lookup = |symbol: &str| {
            ids_by_symbol
                .get(symbol)
                .copied()
                .ok_or_else(|| invalid(format!("BPE merge references missing token {symbol:?}")))
        };
        let result = format!("{left}{right}");
        let left_id = lookup(left)?;
        let right_id = lookup(right)?;
        let result_id = lookup(&result)?;
        let tokens = &builder.model.internal_tokens;
        if !tokens[result_id as usize]
            .bytes
            .iter()
            .copied()
            .eq(tokens[left_id as usize]
                .bytes
                .iter()
                .chain(&tokens[right_id as usize].bytes)
                .copied())
        {
            return Err(invalid(format!(
                "BPE merge {left:?} + {right:?} yields inconsistent token bytes"
            )));
        }
        builder.push_merge(left_id, right_id, result_id);
    }
    builder.finish()
}

pub(super) fn build_export(model: &Model, added_tokens: bool) -> Result<BpeExport, KototokError> {
    let kept = kept_merges(model)?;
    let mut canonical = canonical_tokens(model);
    let mut special_names = FxHashMap::default();
    for (content, id) in &model.special_tokens {
        if content.is_empty() {
            return Err(invalid("empty special token"));
        }
        if model.internal_tokens.iter().any(|token| {
            token.kind != TokenKind::Special
                && token.final_id.is_some_and(|final_id| final_id != *id)
                && token.bytes == content.as_bytes()
        }) {
            return Err(invalid(format!(
                "special token {content:?} conflicts with an ordinary vocabulary token"
            )));
        }
        canonical
            .entry(content.as_bytes())
            .and_modify(|existing| *existing = (*existing).min(*id))
            .or_insert(*id);
        if added_tokens {
            special_names.insert(content.as_bytes(), content.as_str());
        }
    }
    let mut vocab = Vocab::new();
    let mut used_ids = FxHashSet::default();
    for (bytes, id) in canonical {
        if bytes.is_empty() {
            return Err(invalid("empty BPE vocabulary token"));
        }
        let symbol = special_names
            .get(bytes)
            .map_or_else(|| encode_byte_level(bytes), |&name| name.to_owned());
        if !used_ids.insert(id) || vocab.insert(symbol, id).is_some() {
            return Err(invalid(
                "BPE vocabulary has conflicting token strings or IDs",
            ));
        }
    }

    // Walk the unused IDs in ascending order. This consumes holes before
    // allocating IDs above the original maximum.
    let mut next_id = 0u64;
    let mut register = |vocab: &mut Vocab, symbol: String| -> Result<(), KototokError> {
        if !symbol.is_ascii() && special_names.contains_key(symbol.as_bytes()) {
            return Err(invalid(format!(
                "special token {symbol:?} conflicts with a ByteLevel token"
            )));
        }
        if let std::collections::btree_map::Entry::Vacant(entry) = vocab.entry(symbol) {
            while let Ok(id) = u32::try_from(next_id) {
                next_id += 1;
                if !used_ids.contains(&id) {
                    entry.insert(id);
                    return Ok(());
                }
            }
            return Err(invalid("BPE vocabulary token IDs exhausted"));
        }
        Ok(())
    };

    for byte in 0..=u8::MAX {
        register(&mut vocab, bytes_to_unicode(byte).to_string())?;
    }
    let mut merges = Vec::new();
    let mut unique = FxHashSet::default();
    let mut emit = |vocab: &Vocab, left: &str, right: &str| {
        if vocab.contains_key(left)
            && vocab.contains_key(right)
            && vocab.contains_key(&format!("{left}{right}"))
        {
            let line = format!("{left} {right}");
            if unique.insert(line.clone()) {
                merges.push(line);
            }
        }
    };

    for token in &model.internal_tokens {
        if model.pretok.byte_mode == ByteMode::Utf8Atomic
            && token.kind == TokenKind::AtomicChar
            && kept.tokens[token.internal_id as usize]
        {
            let bytes = &token.bytes;
            for end in 2..=bytes.len() {
                let left = encode_byte_level(&bytes[..end - 1]);
                let right = bytes_to_unicode(bytes[end - 1]).to_string();
                register(&mut vocab, format!("{left}{right}"))?;
                emit(&vocab, &left, &right);
            }
        }
    }

    for (rank, event) in model.events.iter().enumerate() {
        if kept.ranks[rank]
            && let VocabEvent::Merge {
                left,
                right,
                result,
            } = *event
        {
            let left = encode_byte_level(&model.internal_tokens[left as usize].bytes);
            let right = encode_byte_level(&model.internal_tokens[right as usize].bytes);
            let result = encode_byte_level(&model.internal_tokens[result as usize].bytes);
            if left.is_empty() || right.is_empty() || format!("{left}{right}") != result {
                return Err(invalid(format!(
                    "BPE merge at rank {rank} yields inconsistent token bytes"
                )));
            }
            for symbol in [&left, &right, &result] {
                register(&mut vocab, symbol.clone())?;
            }
            emit(&vocab, &left, &right);
        }
    }

    if vocab
        .values()
        .copied()
        .max()
        .is_some_and(|id| id as usize >= vocab.len())
    {
        let mut ordered: Vec<_> = vocab.values_mut().collect();
        ordered.sort_unstable_by_key(|id| **id);
        for (dense_id, id) in ordered.into_iter().enumerate() {
            *id = u32::try_from(dense_id).map_err(|_| invalid("too many BPE tokens"))?;
        }
    }
    let mut added: Vec<_> = special_names
        .into_values()
        .map(|content| AddedToken {
            id: vocab[content],
            properties: AddedTokenProperties {
                content: content.into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            },
        })
        .collect();
    added.sort_unstable_by_key(|token| token.id);
    Ok(BpeExport {
        vocab,
        merges,
        added_tokens: added,
    })
}

struct ReachableMerges {
    tokens: Vec<bool>,
    ranks: Vec<bool>,
}

fn kept_merges(model: &Model) -> Result<ReachableMerges, KototokError> {
    let size = model.internal_tokens.len();
    let mut prune_rank = vec![usize::MAX; size];
    let mut tokens = Vec::with_capacity(size);
    for (iid, token) in model.internal_tokens.iter().enumerate() {
        if token.internal_id as usize != iid {
            return Err(invalid(format!(
                "internal token ID mismatch at index {iid}"
            )));
        }
        tokens.push(token.final_id.is_some());
    }
    for (rank, event) in model.events.iter().enumerate() {
        if let VocabEvent::Prune { target, .. } = event {
            let entry = prune_rank
                .get_mut(*target as usize)
                .ok_or_else(|| invalid(format!("invalid prune target at rank {rank}")))?;
            *entry = (*entry).min(rank);
        }
    }

    let mut ranks = vec![false; model.events.len()];
    for (rank, event) in model.events.iter().enumerate() {
        if let VocabEvent::Merge {
            left,
            right,
            result,
        } = *event
        {
            let (left, right, result) = (left as usize, right as usize, result as usize);
            if [left, right, result].into_iter().any(|id| id >= size) {
                return Err(invalid(format!("invalid merge token ID at rank {rank}")));
            }
            if prune_rank[result] == rank + 1 {
                tokens[result] = false;
            } else {
                ranks[rank] = prune_rank[left] >= rank && prune_rank[right] >= rank;
            }
        }
    }
    // Producers precede their consumers in the native chronological engine.
    for (rank, event) in model.events.iter().enumerate().rev() {
        if ranks[rank]
            && let VocabEvent::Merge {
                left,
                right,
                result,
            } = *event
        {
            ranks[rank] = tokens[result as usize];
            if ranks[rank] {
                tokens[left as usize] = true;
                tokens[right as usize] = true;
            }
        }
    }
    Ok(ReachableMerges { tokens, ranks })
}
