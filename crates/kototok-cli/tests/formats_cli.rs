use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const TEXT: &str = "新しいトークナイザーの設計を調べる。hello world 123";

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, Ordering::SeqCst);
        let path = env::temp_dir().join(format!(
            "kototok-cli-formats-{label}-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run_kototok(args: Vec<String>) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_kototok"))
        .args(&args)
        .output()
        .expect("spawn kototok");
    assert!(
        output.status.success(),
        "kototok {:?} failed\nstdout:\n{}\nstderr:\n{}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn stdout(output: Output) -> String {
    String::from_utf8(output.stdout).unwrap()
}

fn path_string(path: &Path) -> String {
    path.to_str().unwrap().to_owned()
}

fn write_corpus(dir: &TestDir) -> PathBuf {
    let path = dir.path().join("corpus.txt");
    let mut corpus = String::new();
    for _ in 0..16 {
        corpus.push_str(TEXT);
        corpus.push('\n');
        corpus.push_str("トークナイザーは日本語とASCIIを分けて学習する。hello world\n");
    }
    fs::write(&path, corpus).unwrap();
    path
}

fn train_model(dir: &TestDir, output: &Path, format: Option<&str>) {
    let corpus = write_corpus(dir);
    let mut args = vec![
        "train".to_owned(),
        "--input".to_owned(),
        path_string(&corpus),
        "--output".to_owned(),
        path_string(output),
        "--vocab-size".to_owned(),
        "320".to_owned(),
        "--threads".to_owned(),
        "1".to_owned(),
    ];
    if let Some(format) = format {
        args.push("--format".to_owned());
        args.push(format.to_owned());
    }
    run_kototok(args);
}

fn encode_text(model: &Path) -> Vec<String> {
    let output = run_kototok(vec![
        "encode".to_owned(),
        "--model".to_owned(),
        path_string(model),
        "--text".to_owned(),
        TEXT.to_owned(),
    ]);
    let ids = stdout(output);
    let ids: Vec<String> = ids.split_whitespace().map(str::to_owned).collect();
    assert!(!ids.is_empty(), "{} produced no token ids", model.display());
    ids
}

fn assert_roundtrip_and_inspect(model: &Path, expected_format: &str) {
    let ids = encode_text(model);
    let decoded = run_kototok(vec![
        "decode".to_owned(),
        "--model".to_owned(),
        path_string(model),
        "--ids".to_owned(),
        ids.join(","),
    ]);
    assert_eq!(stdout(decoded).trim_end_matches('\n'), TEXT);

    let inspect = run_kototok(vec![
        "inspect".to_owned(),
        "--model".to_owned(),
        path_string(model),
    ]);
    let inspect = stdout(inspect);
    let expected_line = format!("format: {expected_format}");
    assert!(
        inspect.lines().any(|line| line.trim() == expected_line),
        "human inspect output lacks `{expected_line}`:\n{inspect}"
    );

    let inspect_json = run_kototok(vec![
        "inspect".to_owned(),
        "--model".to_owned(),
        path_string(model),
        "--json".to_owned(),
    ]);
    let audit: serde_json::Value = serde_json::from_str(&stdout(inspect_json)).unwrap();
    assert!(audit.get("alive_vocab_size").is_some());
    assert!(audit.get("format").is_none());
}

#[test]
fn export_all_formats_roundtrip_and_convert_alias() {
    let dir = TestDir::new("exports");
    let model = dir.path().join("model.ktok");
    train_model(&dir, &model, None);

    let hf = dir.path().join("tokenizer.json");
    run_kototok(vec![
        "export".to_owned(),
        "--model".to_owned(),
        path_string(&model),
        "--output".to_owned(),
        path_string(&hf),
    ]);
    assert!(hf.is_file());

    let hf_dir = dir.path().join("hf-dir");
    run_kototok(vec![
        "export".to_owned(),
        "--model".to_owned(),
        path_string(&model),
        "--output".to_owned(),
        path_string(&hf_dir),
        "--format".to_owned(),
        "hf-dir".to_owned(),
    ]);
    assert!(hf_dir.join("tokenizer.json").is_file());
    assert!(hf_dir.join("tokenizer_config.json").is_file());

    let tiktoken = dir.path().join("model.tiktoken");
    run_kototok(vec![
        "export".to_owned(),
        "--model".to_owned(),
        path_string(&model),
        "--output".to_owned(),
        path_string(&tiktoken),
    ]);
    assert!(tiktoken.is_file());

    let bpe = dir.path().join("bpe");
    run_kototok(vec![
        "export".to_owned(),
        "--model".to_owned(),
        path_string(&model),
        "--output".to_owned(),
        path_string(&bpe),
        "--format".to_owned(),
        "bpe".to_owned(),
    ]);
    assert!(bpe.join("vocab.json").is_file());
    assert!(bpe.join("merges.txt").is_file());

    let json = dir.path().join("model.json");
    run_kototok(vec![
        "convert".to_owned(),
        "--model".to_owned(),
        path_string(&model),
        "--output".to_owned(),
        path_string(&json),
    ]);
    assert!(json.is_file());

    for (exported, expected_format) in [
        (&model, "ktok"),
        (&hf, "hf"),
        (&hf_dir, "hf-dir"),
        (&tiktoken, "tiktoken"),
        (&bpe, "bpe"),
        (&json, "json"),
    ] {
        assert_roundtrip_and_inspect(exported, expected_format);
    }

    let compress = dir.path().join("compress.hf.json");
    run_kototok(vec![
        "export".to_owned(),
        "--model".to_owned(),
        path_string(&model),
        "--output".to_owned(),
        path_string(&compress),
        "--mode".to_owned(),
        "compress".to_owned(),
    ]);
    assert!(compress.is_file());
    assert_roundtrip_and_inspect(&compress, "hf");

    let tagged = dir.path().join("tagged.tiktoken");
    run_kototok(vec![
        "export".to_owned(),
        "--model".to_owned(),
        path_string(&model),
        "--output".to_owned(),
        path_string(&tagged),
        "--tags".to_owned(),
        "ja,casual,kozh,okuri".to_owned(),
    ]);
    assert!(tagged.is_file());
    assert_roundtrip_and_inspect(&tagged, "tiktoken");
}

#[test]
fn train_infers_hf_tokenizer_json_from_output_name() {
    let dir = TestDir::new("train-hf");
    let output = dir.path().join("tokenizer.json");
    train_model(&dir, &output, None);

    let text = fs::read_to_string(&output).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["model"]["type"], "BPE");
    assert!(value["model"]["vocab"].is_object());
    assert!(value.get("internal_tokens").is_none());
    assert_roundtrip_and_inspect(&output, "hf");
}

#[test]
fn train_explicit_tiktoken_format() {
    let dir = TestDir::new("train-tiktoken");
    let output = dir.path().join("model.tiktoken");
    train_model(&dir, &output, Some("tiktoken"));

    let text = fs::read_to_string(&output).unwrap();
    assert!(
        text.lines()
            .next()
            .is_some_and(|line| line.split_whitespace().count() == 2)
    );
    assert_roundtrip_and_inspect(&output, "tiktoken");
}
