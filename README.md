# kototok — AsymScript-BPE

日本語・英語・コード混在入力向けのサブワードトークナイザー。
辞書不要・学習済み語彙の不正 UTF-8 ゼロ・最悪 O(N) エンコードを Rust で実現する。

形式仕様・不変条件・計算量証明: [docs/ALGORITHM.md](docs/ALGORITHM.md)

## 仕組み（3 つの柱）

1. **非対称文字種遷移 DFA 前処理** — `Kanji→Hiragana`（送り仮名）は継続、`Hiragana→Kanji`（助詞＋次の名詞）は分割。`食べる` を `食 | べる` に割らず、`に会議` を `に | 会議` で切る。
2. **UTF-8 原子性 + ACSG + PickyBPE の二段階学習** — 学習語彙の不正 UTF-8 混入率 0%、中間トークンの死蔵を抑制。
3. **DFA ベースの完全可逆エンコード** — 正規表現バックトラックを使わず最悪 O(N)。敵対的 1 MiB 入力でも遅延しない。

## ワークスペース

| クレート | 役割 |
| --- | --- |
| `crates/kototok-core` | DFA 前処理・エンコード/デコード・モデル定義（32 テスト） |
| `crates/kototok-train` | Rayon 並列トレーナー（Phase-1 BPE + Phase-2 チャンク結合） |
| `crates/kototok-cli` | `kototok` バイナリ（`train` / `encode` / `decode` / `inspect` / `viz` / `bench`） |
| `crates/kototok-py` | PyO3 バインディング（`import kototok` で encode/decode/train） |

## クイックスタート

```bash
cargo build --release          # バイナリ: target/release/kototok
cargo test --workspace         # 全テスト（43 件）
cargo clippy --workspace       # 警告ゼロ

# 学習 → 検査 → エンコード
kototok train  --input corpus.txt --output model.ktok --vocab-size 4096 --preset asym-full
kototok inspect --model model.ktok --json
echo "日本語のテキスト" | kototok encode --model model.ktok --input /dev/stdin --json
kototok bench  --model model.ktok --input corpus.txt --threads 1 --iters 5 --json
```

### Python から使う

```bash
PYO3_PYTHON=$(which python3) cargo build --release -p kototok-py
cp target/release/libkototok.so pylib/kototok.so   # sys.path 上に置く
# または maturin: cd crates/kototok-py && maturin develop --release
```

```python
import kototok

tok = kototok.train("corpus.txt", output="model.ktok", vocab_size=4096, preset="asym-full")
# または tok = kototok.Tokenizer.load("model.ktok")
ids    = tok.encode("新しいトークナイザーの設計を調べる。")
pieces = tok.pieces("新しいトークナイザーの設計を調べる。")
text   = tok.decode(ids)          # 完全可逆ラウンドトリップ
batch  = tok.encode_batch(["東京で会議が始まる。", "hello world"])
```

## ベンチマーク（10 モデル）

再現手順:

```bash
cd eval && uv sync                       # または pip install -r <(uv pip compile pyproject.toml)
python corpus_builder.py                 # eval/artifacts/data に 3.1MB コーパス生成（初回のみ）
python run_eval.py --vocab-size 4096     # 全 10 モデル評価 → artifacts/{results.json,report.md}
python make_chart.py                     # artifacts/chart.html

# 大規模版: train コーパス x4 (12.4MB) + vocab_size 32768
python run_eval.py --vocab-size 32768 --output-dir artifacts32k --train-scale 4
python make_chart.py artifacts32k

# 実データ版: jawiki+Gutenberg+OSSコード 22.3MB + vocab_size 32768
python realdata_builder.py                  # eval/artifacts_real/data に実テキスト DL
python run_eval.py --vocab-size 32768 --output-dir artifacts_real
python make_chart.py artifacts_real
```

### 実データ評価（vocab 32768、ja.wikipedia + Gutenberg + OSS code 22.3MB）

`kototok:asym-full` vs `hf-bpe` vs `tiktoken:cl100k_base`\*:

| 指標 | asym-full | hf-bpe | cl100k* |
| --- | ---: | ---: | ---: |
| JA 圧縮 (chars/token) | 1.784 | 1.784 | 0.909 |
| 形態素境界 F1 | **0.8268** | 0.7389 | 0.5968 |
| 不正 UTF-8 学習トークン率 | **0.00%** | 2.17% | 0.65% |
| 送り仮名保持率 | 35.51% | 38.37% | 0.00% |
| JA スループット | **52.4 MB/s** | 3.8 | 13.9 |
| 敵対的 1 MiB レイテンシ | **10.3 ms** | 344 | 78 |

実データでの読み方:

- **圧縮率は hf-bpe と同等** — 合成コーパスで見えた優位はテンプレ反復の産物。kototok の売りは圧縮ではなく、境界精度・UTF-8 妥当性・速度。
- **境界 F1 で +9pt**（0.827 vs 0.739）、名詞+助詞の過結合率も 9.5% vs 19.6% で半減。
- **不正 UTF-8 トークン 0%** は 32k 語彙でも維持（hf-bpe は 705 個・2.17%）。
- 送り仮名保持では hf-bpe/unigram が僅差で上 — ACSG 無効化 (`asym-no-acsg`) なら 46.5% だが名詞+助詞過結合が 38% に悪化。`asym-full` はそのトレードオフ点。
- 実コーパスでは 6 プリセット全て alive vocab = 32768 を充足（合成では 27528 上限）。

### 合成データ評価（参考）

| 指標 | vocab 4k: asym-full / hf-bpe | vocab 32k: asym-full / hf-bpe |
| --- | --- | --- |
| JA 圧縮 (chars/token) | **4.608** / 4.343 | **5.696** / 5.529 |
| 不正 UTF-8 学習トークン率 | **0.00%** / 23.78% | **0.00%** / 3.55% |
| 送り仮名保持率 | **48.16%** / 20.00% | **48.16%** / 39.18% |
| 敵対的 1 MiB レイテンシ | **11.4 ms** / 552 | **15.2 ms** / 385 |

\* tiktoken は 100k–200k 語彙の外部参照。同サイズ同士の直接比較は `hf-*` 行を参照。
全結果: [artifacts](eval/artifacts/report.md)（4k合成）・[artifacts32k](eval/artifacts32k/report.md)（32k合成）・[artifacts_real](eval/artifacts_real/report.md)（32k実データ）
可視化: [chart 4k](eval/artifacts/chart.html)・[chart 32k](eval/artifacts32k/chart.html)・[chart real](eval/artifacts_real/chart.html)

## ライセンス

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) のデュアルライセンス。
