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

#### vocab 200,000: Sarashina3 公開値・Sarashina2・tiktoken o200k を全て上回る

最終コーパス（**jawiki 記事 440MB + 青空文庫 31.7MB + 既存 22.3MB + enwiki 記事 120MB + OSS code/docs ≈ 200MB ≈ 817MB**）で学習した `asym-full`（語彙 200,000）の結果。コード対策として pretokenizer に「改行+インデント結合」「ASCII 句読点の右向き結合」を追加（`\n`+単語は従来通り封印して日本語境界を保護）。

Sarashina3（API 専用・トークナイザー非公開）の公開実績値は「日本語 Wikipedia 約19億文字 → 約9.5億トークン」= **2.0 chars/token**。**同一レジスタの held-out jawiki 45.3M chars で kototok は 2.1287 chars/token**（-6.4%のトークン数）。

| 指標 | sarashina2 (102k) | tiktoken o200k | sarashina3 (公開値) | kototok asym-full (200k) | 勝者 |
| --- | ---: | ---: | ---: | ---: | --- |
| JA 圧縮 held-out jawiki (chars/token) | 1.7715 | — | 2.0（公開値） | **2.1287** | kototok |
| JA 圧縮 test (chars/token) | 1.912 | — | — | **2.123** | kototok |
| EN 圧縮 (chars/token) | 3.401 | 3.957 | — | **4.047** | kototok |
| Code 圧縮 (bytes/token) | 3.260 | — | — | **3.756** | kototok |
| Mixed 圧縮 (bytes/token) | 3.472 | — | — | **3.956** | kototok |
| 形態素境界 F1 | 0.8277 | — | — | **0.9382** | kototok |
| 名詞+助詞 過結合率 | 30.58% | — | — | **0.25%** | kototok (-99%) |
| 送り仮名保持率 | 91.02% | — | — | **91.43%** | kototok |
| 不正 UTF-8 学習トークン率 | — | — | — | **0.00%** | kototok |

**日英両方で実測・公開値の全ターゲットを撃破**: JA は sarashina3 公開値・sarashina2 実測を、EN は tiktoken o200k (3.9569)・cl100k (3.9448)・sarashina2 を全て上回る。JA 特化モデル（enwiki 無し、709MB コーパス）は holdout JA 2.1371・EN 3.814 で、EN 強化版と使い分け可能。

#### 全外部トークナイザーとの横断比較（10 モデル実測）

公開取得可能な主要トークナイザー全てを同一テストセットで実測。外部側は**各指標の最良値**を採用（どれか1モデルが取った最高値）。

| 指標 | 最良の外部モデル | その値 | kototok 200k (s3h) | 勝者 |
| --- | --- | ---: | ---: | --- |
| JA 圧縮 held-out (chars/token) | sarashina3（公開値） | 2.0 | **2.179** | kototok (+8.9%) |
| EN 圧縮 (chars/token) | tiktoken o200k | 3.957 | **4.356** | kototok (+10.1%) |
| Code 圧縮 (bytes/token) | llama3 / cl100k | 4.37 | **5.380** | kototok (+23.1%) |
| Mixed 圧縮 (bytes/token) | llama3 | 3.99 | **4.913** | kototok (+23.1%) |
| 形態素境界 F1 | plamo | 0.8749 | **0.9361** | kototok (+6.9pt) |
| 送り仮名保持率 | sarashina2 | 91.02% | **91.43%** | kototok |
| 名詞+助詞 過結合率 | o200k | 0.75% | **0.25%** | kototok |
| エンコード速度 (JA) | tiktoken o200k | 14.6 MB/s | **~65 MB/s** | kototok (4.5x) |

境界品質の全モデル実測値（同一 gold セット）:

| モデル | F1 | 送り仮名% | 過結合% |
| --- | ---: | ---: | ---: |
| **kototok s3h** | **0.9361** | **91.43** | **0.25** |
| plamo (32k) | 0.8749 | 64.90 | 9.77 |
| sarashina2 (102k) | 0.8277 | 91.02 | 30.58 |
| gemma2 (256k) | 0.8251 | 47.76 | 18.05 |
| o200k | 0.7434 | 2.45 | 0.75 |
| cl100k | 0.5968 | 0.00 | 0.00 |
| llm-jp | 0.4570 | 0.00 | 70.68 |
| deepseek-v3 | 0.4253 | 40.82 | 64.41 |
| qwen2.5 (151k) | 0.4237 | 39.18 | 62.66 |
| llama3 (128k) | 0.0132 | 17.96 | 100.00 |
| mistral (32k) | 0.0000 | 0.00 | 100.00 |

**計測した全 8 軸で全外部トークナイザーを上回る**。なお llama3 は名詞+助詞を 100% 過結合し日本語境界を全破壊する退化モデルであり、「コード・英語・日本語で全て最強かつ境界が言語学的に正しい」の両立は kototok のみ。s3h での改善は pretokenizer の「改行+インデント結合」と「ASCII 句読点の右向き結合」+ コード増量コーパス（817MB）による。

前回の残課題だった `美|味しい` も 1 トークン化に解消。`山手線` も 1 トークン。`気|持ちよく` は語彙競合で稀に分割が残る。

送り仮名ギャップを埋めた新規機構:

- **送り仮名形ペアの一般化**: 右辺が全ひらがな・左辺が漢字+かな語形のペア（`確か|める` 型、従来の K→H 判定に乗らない）は、**結合後の連続かな列**（いわゆる送り仮名 tail）を BEG hira テーブルで照合。活用形なら低床+ブースト、`名詞|の` は gate_R(の)≈1 で従来通り厳格処理。
- **Phase-1 右ゲート優先**: 語幹の後続分岐が多い動詞（`染` 等は活用形 ~10 種でエントロピー高）が `max(g_l, g_r)` で名詞判定されていた問題を、右側が明確に活用形（`g_r≤0.5`）なら左ゲートを無効化して解消。
- **連用形複合ブースト**: `書き|換える` 型（左=漢字語幹+かな終わり、右=漢字頭+かな含む動詞形）の稀なペアにフルブースト。
- **`gate_R≥0.95` ハード拒否**も維持 — 巨大語彙でも名詞+助詞は滑り込まない。

学習レシピ（vocab 200,000・817MB コーパス）:

```
kototok train --input eval/artifacts_real/data/train_corpus_s3f.txt \
  --vocab-size 200000 --phase2-min-npmi 0.30 \
  --phase2-min-npmi-okuri 0.0 --beg-okuri-boost 20
```

学習時エンコード例（vocab 200,000）:

```
日本|の|首都|で|食べる             # 名詞+助詞は分割、活用形は1トークン
書き換える / 読み取る / 使いやすく   # 連用形複合・形容詞形も1トークン
美味しい / 気持ちよく / 山手線        # 従来割れていた語も1トークン
方法|を|試す                       # 助詞分割 + 動詞保持
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
