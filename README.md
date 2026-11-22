# kototok — AsymScript-BPE

日本語・英語・コード混在入力向けのサブワードトークナイザー。
辞書不要・学習済み語彙の不正 UTF-8 ゼロ・最悪 O(N) エンコードを Rust で実現する。

形式仕様・不変条件・計算量証明: [docs/ALGORITHM.md](docs/ALGORITHM.md)

## 仕組み（4 つの柱）

1. **非対称文字種遷移 DFA 前処理** — `Kanji→Hiragana`（送り仮名）は継続、`Hiragana→Kanji`（助詞＋次の名詞）は分割。`食べる` を `食 | べる` に割らず、`に会議` を `に | 会議` で切る。
2. **BEG（Boundary-Entropy Gate）— 新規の形態境界推定** — `Kanji→Hiragana` 結合には「語幹＋送り仮名」（良）と「名詞＋助詞」（悪）が混在する。BEG はコーパス統計だけで両者を見分ける: 左側漢字連の右分岐エントロピー（語幹は活用形が閉じて低い、名詞は助詞が開いて高い）× 右側ひらがな列の左分岐多様性（助詞は数万の名詞に付く、送り仮名は少数の語幹だけ）。ゲート値で NPMI ペナルティの指数を変調し、辞書なしで `食+べる` は通し `日本+の` を止める。
3. **UTF-8 原子性 + PickyBPE の二段階学習** — 学習語彙の不正 UTF-8 混入率 0%、中間トークンの死蔵を抑制。
4. **DFA ベースの完全可逆エンコード** — 正規表現バックトラックを使わず最悪 O(N)。敵対的 1 MiB 入力でも遅延しない。

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

`kototok:asym-full`（BEG 有効）vs `hf-bpe` vs `tiktoken:cl100k_base`\*:

| 指標 | asym-full | hf-bpe | cl100k* |
| --- | ---: | ---: | ---: |
| JA 圧縮 (chars/token) | 1.751 | 1.784 | 0.909 |
| EN 圧縮 (chars/token) | **4.014** | 3.490 | 3.945 |
| 形態素境界 F1 | **0.8864** | 0.7389 | 0.5968 |
| 名詞+助詞 過結合率 | **0.25%** | 19.55% | 0.00% |
| 送り仮名保持率 | **53.88%** | 38.37% | 0.00% |
| 不正 UTF-8 学習トークン率 | **0.00%** | 2.17% | 0.65% |
| JA スループット | **41.8 MB/s** | 2.7 | 15.6 |
| 敵対的 1 MiB レイテンシ | **12.4 ms** | 430 | 69 |

実データでの読み方:

- **BEG が「送り仮名保持 × 助詞分割」の両立を実現**。従来の一様 NPMI ペナルティ（ACSG β）では送り仮名を残すと名詞+助詞過結合が 38% に悪化した。BEG は両側の分岐統計で対象を絞り、過結合を **0.25%** まで潰しながら送り仮名 53.9% を保持。悪い結合を捨てた分、語幹+送り仮名が語彙枠を取れるようになったため。
- **境界 F1 は全モデル中最高**（0.8864、hf-bpe +14.7pt、hf-unigram +6.0pt）。
- **クラス別 Phase-2 で EN 圧縮も全モデル最高**（4.014、語彙 3〜6 倍の cl100k 3.945・o200k 3.957 も超過）。純 ASCII ペアは緩い NPMI 床（0.10）で superword 化しつつ、ASCII 枠を Phase-2 予算の 30% に制限するクオータで JA superword 枠を保護 — JA/EN 両方の圧縮と品質が同時改善した。
- JA 圧縮は hf-bpe に **1.8% 差まで縮小**（1.751 vs 1.784）— 名詞+助詞を意図的に割く設計の代償は残るが、形態素境界の正しさとのトレードオフとしては最小限。
- **不正 UTF-8 トークン 0%** は 32k 語彙でも維持（hf-bpe は 2.17%）。
- tiktoken の過結合 0% は「日本語をほぼ結合しない」（送り仮名 0%）の退化点であり、実用とは別物。

#### vocab 102,400: Sarashina2 との直接対決

同じ語彙サイズ（102,400）で学習した `asym-full`（`--phase2-min-npmi 0.35` + BEG）を、日本語特化の代表格 **Sarashina2**（SentencePiece Unigram、~1T トークンコーパス学習）と比較。kototok の学習コーパスは 22.3MB。

| 指標 | sarashina2 | kototok asym-full | 勝者 |
| --- | ---: | ---: | --- |
| JA 圧縮 (chars/token) | 1.912 | **1.972** | kototok (+3%) |
| EN 圧縮 (chars/token) | 3.401 | **4.506** | kototok (+32%) |
| Code 圧縮 | 3.260 | **3.655** | kototok (+12%) |
| Mixed 圧縮 | 3.075 | **3.406** | kototok (+11%) |
| 形態素境界 F1 | 0.8277 | **0.9081** | kototok (+8pt) |
| 名詞+助詞 過結合率 | 30.58% | **0.75%** | kototok (-97%) |
| 送り仮名保持率 | **91.02%** | 72.24% | sarashina |

7 項目中 6 項目で勝利。唯一の敗北である送り仮名保持率はコーパス規模差（1T vs 22MB）由来 — Sarashina は全活用形を丸ごと語彙化している。それでも `食べた|食べて|食べない` 等は 1 トークン化される。

ポイント: kototok の Phase-2 BEG は **gate_R（右側ひらがな列の多様性）のみ** を適用する。`日本+の` のような名詞+助詞ペアは `gate_R(の)≈1` で拒否され、さらに `gate_R≥0.95` のペアは**ハード拒否**されるため、巨大語彙でも高 NPMI の名詞+助詞が滑り込まない。

学習時エンコード例（実データ語彙、vocab 32768）:

```
私|は|リン|ゴ|を|食べる|。          # 語幹+送り仮名は保持、助詞は分割
日本|の|首都|は|東京|である|。      # 名詞+助詞の過結合を抑止
彼|は|勉|強|する|。                # サ変動詞の「する」は独立トークン
新しい|方法|を|試|す|。
```

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
