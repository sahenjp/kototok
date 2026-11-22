//! Integration tests: train all six presets on a mixed JP/EN/code corpus,
//! then verify serialization round-trip, lossless encode/decode, and the
//! UTF-8 completeness invariant.

use std::path::PathBuf;

use kototok_core::{ByteMode, Model, Tokenizer};

use crate::{TrainPreset, train};

/// Deterministic ~35KB mixed corpus. Fixed template lines guarantee high-IoS
/// intermediates (so PickyBPE prunes), while per-line variation supplies a
/// diverse pair distribution.
fn fixture_corpus() -> String {
    let mut s = String::new();
    for i in 0..60u32 {
        s.push_str(&format!("// モジュール {i}: 高速トークナイザー実装\n"));
        s.push_str("use std::collections::HashMap;\n");
        s.push_str(&format!(
            "fn process_{i}(data: &[u8]) -> Result<Vec<u8>, Error> {{\n"
        ));
        s.push_str("    let mut table = HashMap::new();\n");
        s.push_str("    for (idx, byte) in data.iter().enumerate() {\n");
        s.push_str("        table.insert(idx, byte.wrapping_add(1));\n");
        s.push_str("    }\n");
        s.push_str("    Ok(table.into_values().collect())\n");
        s.push_str("}\n\n");
        // Identical repeated sentences -> deterministic continuation -> prunes.
        s.push_str("株式会社サンプルのシステムは安定稼働を続けている。\n");
        s.push_str("開発者が日本語のテキストを効率的に処理するための仕組みを設計した。\n");
        s.push_str(
            "送り仮名を含む動詞、例えば食べる、調べる、比べるは一つのチャンクとして扱われる。\n",
        );
        s.push_str(&format!(
            "日本語の文章 {i}: 新しいトークナイザーは高速に動作する。\n"
        ));
        s.push_str("The quick brown fox jumps over the lazy dog repeatedly.\n");
        s.push_str(
            "Tokenization efficiency determines inference latency in production systems.\n\n",
        );
    }
    assert!(s.len() >= 30 * 1024);
    s
}

fn sample_texts() -> Vec<&'static str> {
    vec![
        "新しいRustのトークナイザーを作る。",
        "日本語の開発者が高速な処理を書く。",
        "The quick brown fox jumps over 42 lazy dogs!",
        "fn main() { let x = vec![1, 2, 3]; println!(\"{x:?}\"); }",
        "絵文字テスト🎉🎊✨のエンコード確認。",
        "鬱薔薇𠮟", // unseen rare kanji -> ByteFallback expansion
        "mixing 日本語 and English in one line 12345.",
        "カタカナ語ーと長音符のテスト。",
    ]
}

fn tmp_path(name: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "kototok_train_test_{}_{}.{ext}",
        std::process::id(),
        name
    ))
}

#[test]
fn all_presets_train_roundtrip_and_audit() {
    let corpus = fixture_corpus();
    for preset in TrainPreset::ALL {
        let cfg = preset.config(2048);
        let model = train(&corpus, &cfg).unwrap();

        // Serialize -> reload -> compile.
        let ktok = tmp_path(preset.name(), "ktok");
        let json = tmp_path(preset.name(), "json");
        model.save_binary(&ktok).unwrap();
        model.save_json(&json).unwrap();
        let reloaded = Model::load_auto(&ktok).unwrap();
        assert_eq!(reloaded, model, "{preset}: binary roundtrip mismatch");
        let reloaded_json = Model::load_auto(&json).unwrap();
        assert_eq!(reloaded_json, model, "{preset}: json roundtrip mismatch");
        let tok = Tokenizer::from_model(reloaded).unwrap();

        // Lossless round-trip, zero PrunedTokenEmitted.
        for text in sample_texts() {
            let ids = tok.encode(text).unwrap_or_else(|e| {
                panic!("{preset}: encode failed for {text:?}: {e}");
            });
            let back = tok.decode(&ids).unwrap();
            assert_eq!(back, text, "{preset}: lossy roundtrip for {text:?}");
        }

        let audit = model.audit_utf8_validity();
        assert!(
            audit.alive_vocab_size <= cfg.vocab_size,
            "{preset}: alive {} exceeds vocab_size {}",
            audit.alive_vocab_size,
            cfg.vocab_size
        );
        if cfg.pretok.byte_mode == ByteMode::Utf8Atomic {
            assert_eq!(
                audit.invalid_utf8_learned_tokens, 0,
                "{preset}: Utf8Atomic produced invalid UTF-8 learned tokens"
            );
        }
        if preset == TrainPreset::AsymFull {
            assert!(
                audit.pruned_tokens > 0,
                "asym-full: expected PickyBPE to prune at least one token"
            );
        }

        std::fs::remove_file(&ktok).ok();
        std::fs::remove_file(&json).ok();
    }
}

#[test]
fn special_tokens_are_present_and_never_merged() {
    let corpus = fixture_corpus();
    let mut cfg = TrainPreset::AsymNoPhase2.config(512);
    cfg.special_tokens = vec!["<|eos|>".to_string(), "<|bos|>".to_string()];
    let model = train(&corpus, &cfg).unwrap();
    assert_eq!(model.special_tokens.len(), 2);
    for (_, id) in &model.special_tokens {
        let tok = &model.internal_tokens[*id as usize];
        assert_eq!(tok.kind, kototok_core::TokenKind::Special);
        // No merge event may consume or produce a special token.
        for ev in &model.events {
            if let kototok_core::VocabEvent::Merge { left, right, .. } = ev {
                assert!(*left != *id && *right != *id);
            }
        }
    }
    let tok = Tokenizer::from_model(model).unwrap();
    assert!(tok.encode("hello <|eos|> world").is_ok());
}

#[test]
fn tiny_vocab_trains_base_only() {
    let corpus = fixture_corpus();
    let cfg = TrainPreset::AsymFull.config(300);
    let model = train(&corpus, &cfg).unwrap();
    let audit = model.audit_utf8_validity();
    assert!(audit.alive_vocab_size <= 300);
    let tok = Tokenizer::from_model(model).unwrap();
    for text in sample_texts() {
        let ids = tok.encode(text).unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }
}

#[test]
fn vocab_too_small_is_an_error() {
    let cfg = TrainPreset::AsymFull.config(128);
    assert!(train("hello", &cfg).is_err());
}

#[test]
fn beg_gate_training_smoke() {
    // Stem `食` is followed only by {べる,べた,べて}; noun `天気` by 8
    // particles. With BEG on, `食` gets gate~0 (free merge) while `天気`
    // gets gate~1 (full penalty).
    let mut s = String::new();
    for _ in 0..80u32 {
        s.push_str("リンゴを食べる。ミカンも食べた。ブドウも食べて。\n");
        s.push_str(
            "天気の話、天気は晴れ、天気が悪い、天気を見る、天気に注意、\
             天気で決まる、天気と予報、天気から分かる。\n",
        );
    }
    let mut cfg = TrainPreset::AsymFull.config(600);
    cfg.beg_strength = 3.0;
    cfg.beg_stem_max = 1.2;
    cfg.beg_noun_min = 2.0;
    let model = train(&s, &cfg).unwrap();
    let audit = model.audit_utf8_validity();
    assert_eq!(audit.invalid_utf8_learned_tokens, 0);
    let tok = Tokenizer::from_model(model).unwrap();
    let text = "ミカンも食べる。天気が悪い。";
    let ids = tok.encode(text).unwrap();
    assert_eq!(tok.decode(&ids).unwrap(), text);
}
