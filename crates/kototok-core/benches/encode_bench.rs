use criterion::{Criterion, black_box, criterion_group, criterion_main};
use kototok_core::{
    InternalToken, Model, ModelMetadata, PreTokenizerConfig, TokenKind, Tokenizer, VocabEvent,
};

fn bench_model() -> Model {
    let mut internal: Vec<InternalToken> = (0..256u32)
        .map(|b| InternalToken {
            internal_id: b,
            bytes: vec![b as u8],
            kind: if b < 128 {
                TokenKind::Ascii
            } else {
                TokenKind::ByteFallback
            },
            final_id: Some(b),
        })
        .collect();
    // Common Japanese chars as atomic tokens plus a few merges.
    let atomics = [
        "あ", "い", "う", "え", "お", "の", "を", "が", "に", "は", "と", "で", "す", "る", "た",
        "日", "本", "語", "開", "発", "者", "高", "速", "処", "理", "書", "く", "新", "し", "ト",
        "ー", "ク", "ナ", "イ", "ザ", "作", "。",
    ];
    for (i, s) in atomics.iter().enumerate() {
        internal.push(InternalToken {
            internal_id: 256 + i as u32,
            bytes: s.as_bytes().to_vec(),
            kind: TokenKind::AtomicChar,
            final_id: Some(256 + i as u32),
        });
    }
    let next = 256 + atomics.len() as u32;
    internal.push(InternalToken {
        internal_id: next,
        bytes: "あい".as_bytes().to_vec(),
        kind: TokenKind::MergedPhase1,
        final_id: Some(next),
    });
    let vocab = internal.iter().filter(|t| t.final_id.is_some()).count();
    Model {
        metadata: ModelMetadata {
            name: "bench".into(),
            version: 1,
            vocab_size: vocab,
            phase1_vocab_size: 256,
            phase2_start_rank: 1,
            pruned_count: 0,
            invalid_utf8_learned_tokens: 0,
            train_notes: String::new(),
        },
        pretok: PreTokenizerConfig::default(),
        special_tokens: vec![],
        internal_tokens: internal,
        events: vec![VocabEvent::Merge {
            left: 256,
            right: 257,
            result: next,
        }],
    }
}

const JA_PROSE: &str = "日本の開発者が高速な処理を書く。新しいトークナイザーを作る。";
const RUST_CODE: &str = "fn main() {\n    let x = 12345;\n    println!(\"{}\", x);\n}\n";

fn bench_encode(c: &mut Criterion) {
    let tok = Tokenizer::from_model(bench_model()).unwrap();
    let long_ja = JA_PROSE.repeat(64);
    let long_code = RUST_CODE.repeat(64);
    let hiragana_1mb = "あ".repeat(1024 * 1024);

    let mut group = c.benchmark_group("encode");
    group.bench_function("japanese_prose", |b| {
        b.iter(|| tok.encode(black_box(&long_ja)).unwrap())
    });
    group.bench_function("rust_code", |b| {
        b.iter(|| tok.encode(black_box(&long_code)).unwrap())
    });
    group.sample_size(10);
    group.bench_function("hiragana_1mb", |b| {
        b.iter(|| tok.encode(black_box(&hiragana_1mb)).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
