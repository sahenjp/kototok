# kototok（AsymScript-BPE）

kototok は、日本語・英語・多言語・コード・数式が混在する入力向けの Rust 製サブワードトークナイザーであり、辞書不要・学習時の不正 UTF-8 学習トークン 0（UTF-8 原子性不変条件）・最悪 O(N) の完全可逆エンコードを実現します。
独自の `.ktok` バイナリに加え、kototok JSON・Hugging Face `tokenizer.json` / ディレクトリ・OpenAI `.tiktoken`・GPT-2 形式（`vocab.json` + `merges.txt`）の全 6 形式を自動で読み込み・書き出しできます。
形式仕様、不変条件、および計算量証明の詳細は [docs/ALGORITHM.md](docs/ALGORITHM.md) を参照してください。専門知識のない読者向けの解説は [docs/slides/index.html](docs/slides/index.html) にあります。

## 主な結果

817MB コーパスで学習した 200k モデルでは、非対称前処理と境界エントロピーゲート（BEG）および外科的語彙修復により、高い日本語形態素境界品質と各言語の圧縮効率を両立しています。
さらに 68 ドメイン向けの `global` プリセット（264k 語彙）では、多言語文字種処理と階層フォールバック Phase-2 により、自然言語・数式・コードの全領域で高い圧縮効率と完全可逆性を達成しています。

- 68 ドメイン多言語・数式・コード評価（11.16MB・完全非重複 held-out）では、フラッグシップの `kototok_global_264k.ktok`（alive vocab 264,498、不正 UTF-8 学習トークン 0）が総トークン数 2,261,563（4.932 bytes/token）を記録し、外部 12 モデル（`Custom Global 264,498`・`HF-Global-264k`・`GPT-4o (o200k)`・`Gemma-3`・`Qwen3.5`・`Gemma-2`・`DeepSeek-V3`・`Inkling-2`・`Qwen3`・`Qwen2.5`・`Llama-3`・`GPT-4 (cl100k)`）すべてに対して全 68 ドメインで勝利しました（延べ 816/816 勝、全 12 モデルのドメイン別最良値を集めた仮想アンサンブルに対しても 68/68 全勝）。`Custom Global 264,498`（2,671,022）に対しては -15.3% 少ないトークン数・4.53 倍のエンコード速度（15.17 MB/s vs 3.35 MB/s）、さらに絵文字・国旗・ZWJ・結合記号等の Unicode ストレステストでも -33.9%（HF 書出版 -35.4%）を達成しています。同一 286MB コーパス条件の対照モデル `kototok-Global-Same`（2,234,457）でも `HF-Global-264k`（2,712,790）比で 17.6% 少ないトークン数に抑えています。

- コード向け前処理拡張版の 200k モデル `kototok 200k (s3h)` は、JA 圧縮（held-out 2.179 chars/token）、EN 圧縮（4.356 chars/token）、Code 圧縮（5.380 bytes/token）、Mixed 圧縮（4.913 bytes/token）、150 文 hand-gold の形態素境界 F1（0.9361）、送り仮名保持率（91.43%）、名詞+助詞 過結合率（0.25%）、エンコード速度（~65 MB/s）の 8 軸で、同一テストセット上で実測した外部モデルの各指標最良値を上回りました（JA 圧縮の参照点である Sarashina3 公開値は別テストセットの目標値として扱います）。
- kototok v2-T（kototok_v2s_200k.ktok）は単一ファイル内にタグ付き層を内蔵し、okuedge 層を含めた strict プロファイルで Sudachi gold 500 文 holdout の境界 F1 0.864・送り仮名保持 88.9%・過結合 0.4% を達成します（独立辞書 UniDic でも境界 F1 0.852・送り仮名 89.0%・過結合 0.5%）。
- 多言語対応の kototok v2-BKO3（kototok_v2_bko3_200k.ktok）は、bytes/token 基準で JA prose 5.79、Korean 未見 wiki 3.75、Chinese 未見 wiki 4.01 の圧縮率を示し、jawiki 64MiB 上の trigram-LM 代理評価では kototok 5 モデルが上位 7 枠中 5 枠を占めます（最良 1.437 bits/byte、sarashina2 と 0.004 bits/byte 差）。

## クイックスタート

### Rust ビルドなしで使う（Hugging Face `tokenizers` / `transformers`）

フラッグシップの `kototok_global_264k` は、Hugging Face 形式ディレクトリ（`models/kototok_global_264k_hf/`、14.6MB）も同梱しています。Rust や専用拡張をビルドせず、標準の Python ライブラリからそのまま読み込めます。

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("models/kototok_global_264k_hf/tokenizer.json")
enc = tok.encode("日本語と English と `fn main()` を同時に処理します。")
print(enc.tokens)
print(tok.decode(enc.ids))  # 完全可逆ラウンドトリップ

# Hugging Face transformers の場合:
# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("models/kototok_global_264k_hf")
```

### `kototok` CLI で学習・変換・エンコードする

```bash
cargo build --release

# 形式を自動判定してエンコード（.ktok / tokenizer.json / hf-dir / .tiktoken / bpe / .json）
./target/release/kototok encode --model models/kototok_global_264k.ktok --text "新しいトークナイザー" --pieces

# 任意モデルを Hugging Face / tiktoken / GPT-2 BPE 形式へ書き出し
./target/release/kototok export --model models/kototok_global_264k.ktok --output out_hf --format hf-dir
./target/release/kototok export --model models/kototok_global_264k.ktok --output out.tiktoken
```

## 手法

### 非対称 DFA 前処理と UTF-8 原子性

前処理の非対称文字種遷移 DFA は、`Kanji→Hiragana`（送り仮名）を同一チャンクに継続して `食べる` が `食 | べる` に割れるのを防ぐ一方、`Hiragana→Kanji`（助詞＋次の名詞）を分割して `に会議` を `に | 会議` で切ります。
コード対策として pretokenizer には「改行+インデント結合」と「ASCII 句読点の右向き結合」を組み込んでおり、`\n`+単語の結合は日本語境界を保護するため封印したままです。
学習時は UTF-8 原子性と PickyBPE による二段階学習で不正 UTF-8 混入率 0% と中間トークンの死蔵抑制を実現し、クラス別 Phase-2 では純 ASCII ペアを緩い NPMI 床（0.10）で結合しつつ ASCII 枠を Phase-2 予算の 30% に制限して日本語 superword 枠（ハード境界内の複合語・隣接チャンク結合枠）を保護します。
エンコードは正規表現バックトラックを使わない DFA ベースで完全可逆かつ最悪 O(N) で動作するため、敵対的 1 MiB 入力でも遅延しません。

### 境界エントロピーゲート（BEG）

`Kanji→Hiragana` 結合には「語幹＋送り仮名」と「名詞＋助詞」が混在し、従来の一様 NPMI ペナルティ（ACSG: Asymmetric Cross-Script Cohesiveness Gating の指数 β）では送り仮名を残すと名詞+助詞過結合が 38% に悪化していました。
BEG は、活用形が閉じてエントロピーの低い語幹と助詞が開いて高い名詞を測る「左側漢字連の右分岐エントロピー」（左側ゲート `g_l`）と、少数の語幹に付く送り仮名と数万の名詞に付く助詞を区別する「右側ひらがな列の左分岐多様性」（右側ゲート `g_r` / `gate_R`、BEG hira テーブルに保持）を組み合わせたゲート値で NPMI ペナルティの指数を変調し、辞書なしで `食+べる` を通し `日本+の` を止めます。

- 送り仮名形ペアの一般化では、右辺が全ひらがなで左辺が漢字+かな語形のペア（`確か|める` 型）に対し、結合後の連続かな列（送り仮名 tail）を BEG hira テーブルで照合して活用形には低床とブーストを与え、`名詞|の` は gate_R(の)≈1 として厳格処理します。
- Phase-1 右ゲート優先では、`染` など活用形が ~10 種あり `max(g_l, g_r)` で名詞判定されていた動詞について、右側が明確な活用形（`g_r≤0.5`）であれば左ゲートを無効化します。
- 連用形複合ブーストでは `書き|換える` 型（左が漢字語幹+かな終わり、右が漢字頭+かなを含む動詞形）の稀なペアにフルブーストを与えます。
- `gate_R≥0.95` ハード拒否も維持しており、語彙サイズが大きくなっても名詞+助詞の過結合を防ぎます。

### 外科的 prune/graft（Aligned BPE-Knockout）

Aligned BPE-Knockout では、学習済みモデルに対して gold 境界を跨ぐ merge を切除（prune）し、さらに観測された分割連鎖を低優先の merge イベントとして末尾追加（graft、1 語あたり最大 8 パターンの文脈依存連鎖を保持する multi-chain graft）することで語彙を修復します。
各要素には先行研究が存在するため、本実装の主な差別化点はタグ単位で着脱できる外科レイヤーとその組合せにあります。

| 先行・関連技術 | 関係 |
| --- | --- |
| BPE-Knockout（Bauwens & Delobelle 2024） | prune（学習済み merge の事後切除）は同系。差分: 本実装は merge 削除だけでなく graft（観測分割連鎖を merge イベントとして末尾追加）で修復し、かつ層タグで着脱可能にした拡張（Aligned BPE-Knockout）。 |
| 語彙拡張 / tokenizer transfer（FVT、継続事前学習の語彙置換系） | graft は「既存語彙の後に低優先 merge を追加」する点で同族。差分: graft 源が gold 境界の観測分割連鎖であり、単語あたり最大 8 パターンの文脈依存連鎖（multi-chain graft）を保持する点。 |
| BPE-dropout（Provilkov et al. 2020）・SentencePiece n-best sampling | 「複数分割の共存」は共通テーマ。差分: 本方式は学習/エンコード時の確率的摂動ではなく、モデルファイルが決定論的な代替 merge 層を内蔵しロード時に層選択する静的機構。 |
| Sudachi / UniDic 等の辞書分割 | gold の出典として利用（修復・評価両方）。モデル自体は辞書非依存で、gold が無い言語では同じ手術は行えない制限あり。 |

### タグ付きポリシー切替（Selectable-Segmentation）

Selectable-Segmentation（v2-T、`kototok_v2s_200k.ktok`）は、`.ktok` の語彙イベント列に出自タグ付き graft レイヤー（`tag_ranges`、serde 互換で新旧モデル相互ロード可）を内蔵し、確率的サンプリングではなく決定論的に言語学的忠実モードと学習 merge のみの素の圧縮モードを切り替える機構です。
`refine --tag ja|casual|kozh|okuri` で各 graft バッチに出自タグを記録すると、実行時に `encode --mode strict`（既定・全レイヤー有効）、`--mode compress`（全 graft 無効）、`--tags kozh,okuri`（任意選択）、および Python の `Tokenizer.load(path, tags=[...])` で層を切り替えられます。
また、`refine --prune-oku-edges` と `--tag okuedge` は、送り仮名語の両端をまたぐ merge をすべて切除しつつ、ランク位置に依存する prune を範囲タグではなく `(target内部ID → tag)` の疎な対応表（`tagged_prunes`）で記録するため、単一ファイルのまま `strict`（全タグ有効、`okuedge` 含む）と `balanced`（`--tags ja,casual,kozh,okuri`）を切り替えられます。

### 多言語混合と事後 graft

多言語対応モデルの `kototok_v2_bko3_200k.ktok`（41.6MB）は、KO/ZH Wikipedia と口語文を学習コーパスへ混入した v2 モデル（776MB、vocab 200k）に対し、JA/口語は Sudachi gold、KO/ZH は whitespace・jieba 由来の gold を用いた Aligned BPE-Knockout（1,328 prune + 64,498 graft。モデル内メタデータ記録では gold 17,536 文に適用）を実施して構築しています。
KO/ZH の学習時混入だけでは未見文の圧縮が 2.88 / 3.49 B/T にとどまるのに対し、gold 境界からの事後 graft によって 3.75 / 4.01 B/T まで伸びており、Aligned BPE-Knockout は gold さえあれば任意言語の語彙を後付けで増設できる機構として機能します。

### 多言語 global プリセットと階層フォールバック Phase-2

68 ドメイン向けの `global` プリセットでは、日本語特化の前処理に加えて多言語文字種処理と省メモリ Phase-2 学習を組み合わせています。

- 多言語文字種 DFA（`multilingual_scripts`）は、Unicode の結合記号・ゼロ幅接合子（インド系・東南アジア系・アラビア系・ヘブライ系等）や拡張ラテン文字（アゼルバイジャン語の `ə` 等）・ギリシャ文字（`µ` 含む）を文字種ごとにまとめて分断を防ぎ、非 CJK の全アルファベット文字種（節区切り空白を含むタイ語・ラーオ語・ミャンマー語・クメール語等を含む）で単一の先頭 ASCII 空白を後続語へ結合します。
- コード・数式・多言語向けの句読点処理（`attach_space_to_punct`）は、ASCII 句読点の前後を Phase-2 のソフト境界に保ちつつ、日本語では単語と句読点の境界を封印したまま隣接する日本語約物同士（`。「` や `！！` 等）のみ結合可能にします。
- 階層フォールバック Phase-2（`phase2_fallback_window = 3`）と Unicode シンボルシード（`seed_unicode_symbols`）により、16 チャンク窓で 2 回以上出現するコード行や数式をそのまま保持しつつ、1 回しか現れない自然言語文は 3 チャンク単位へ分割して反復句を集計します。あわせて絵文字・国旗（Regional Indicator）・ZWJ/異体字セレクタ・結合ダイアクリティカルマーク・罫線・数式記号の主要 17 Unicode ブロックを `AtomicChar` として事前確保し、稀な絵文字や Zalgo テキストでの 4 バイトフォールバック膨張を防ぎます。Phase-1/Phase-2 の状態寿命分離とインプレース更新により、264k 語彙の学習を 4 スレッドで 286MB 約 36〜50 秒（ピーク RSS 1.86 GiB・スワップ 0 KiB）、1.03GB コーパスでも 104.1 秒で完了できます。

### 汎用フォーマット入出力と相互運用

kototok は独自バイナリの `.ktok` に限らず、主要なトークナイザー形式を含む全 6 形式の読み込み・書き出しに対応します。
読み込み時はファイル内容やディレクトリ構成から形式を自動判定し、書き出し時は出力パスから形式を推定します（CLI の `--format` や Python の `format=` で明示指定も可能）。

| 形式指定 | 代表ファイル・構成 | 読込 / 書出 | 外部ランタイム互換性と特徴 |
| --- | --- | :---: | --- |
| `ktok` | `*.ktok` | ○ / ○ | kototok 標準バイナリ。PickyBPE 剪定履歴・タグ付き外科層（`tag_ranges` / `tagged_prunes`）を完全保持 |
| `json` | `*.json`（kototok スキーマ） | ○ / ○ | `ktok` と等価な JSON 表現。差分確認やモデル検査向け |
| `hf` | `tokenizer.json` | ○ / ○ | Hugging Face `tokenizers` 互換。剪定済み祖先を補完した密 ID（`0..V-1`）と多言語 `pre_tokenizer` 正規表現を出力 |
| `hf-dir` | `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` | ○ / ○ | `transformers`（`AutoTokenizer.from_pretrained`）・vLLM 等から Rust ビルドなしで直接読込可能（標準配布推奨） |
| `tiktoken` | `*.tiktoken` | ○ / ○ | OpenAI `tiktoken` 形式（base64 + rank）。仕様上 `pre_tokenizer` 正規表現を持たず、kototok での読込時は標準 GPT モードとして扱う |
| `bpe` | `vocab.json` + `merges.txt` | ○ / ○ | GPT-2 互換の ByteLevel BPE 2 ファイル形式 |

Hugging Face 形式（`hf` / `hf-dir`）の書き出しでは `pre_tokenizer` 設定に `AsymScript`・`StrictScript`・`StandardGpt` と `global` 拡張フラグを反映するため、Hugging Face `tokenizers` ランタイムおよび kototok 再読込の両方で動作を再現できます。

## 評価

### 評価設計とデータリーク監査

- 境界指標は (1) 人手作成の 150 文（held-out 分割 50 文）と (2) Sudachi 自動生成の jawiki 3000 文（surgery 2500 / held-out 500）の 2 種類で計測しており、Sudachi gold のほうが細かく分割するため、その F1 スケールは低く、hand-gold の数値とは直接比較できません。
- 外部比較用の境界指標は 150 行の hand gold 全体で算出しており、kototok の refined モデルについてはこれに加えて held-out スコアを報告しています（報告 F1 は held-out 部分のみ）。
- Sarashina3 の公開値 2.0 chars/token は非公開テストセットに基づくため同一条件ベンチマークではなく目標値として扱い、Casual/SNS テキスト（v2-BKO3 3.97 B/T vs gemma2 3.72）は数行の小さな合成エッジセット上の方向性の指標にとどまります。
- v2-BKO3 の境界指標は同じ生成レシピで新たに再生成した別の Sudachi holdout（未見 500 行）で計測しているため、スケールは比較可能ですが後述の Matched held-out comparison 表（s3o-BKO2 等）と行単位で同一ではありません。
- 2026-11 のリーク監査では、主要レイヤー（`ja`/`casual`/`kozh`/`okuri`/`okuedge`）への適用記録があるすべての gold ファイルで 500 行 holdout との重複がゼロ（完全一致および先頭 60 文字一致）であり、holdout を含む分割前の中間集合 `gold_ja_v2`（3000 行）は refine 入力として未記録であるうえ、`gold_ja_v2` を入力に使わずに再構築した別系統の学習パスでも同等のスコア（balanced F1 0.865 / oku 86.0%）を確認済みです。

### 圧縮効率の比較

#### 32k 実データと合成データ

実データ評価（vocab 32768、ja.wikipedia + Gutenberg + OSS code 22.3MB）における `kototok:asym-full`（BEG 有効）、`hf-bpe`、および `tiktoken:cl100k_base`\* の結果を示します。

| 指標 | asym-full | hf-bpe | cl100k* |
| --- | ---: | ---: | ---: |
| JA 圧縮 (chars/token) | 1.754 | 1.784 | 0.909 |
| EN 圧縮 (chars/token) | **4.340** | 3.490 | 3.945 |
| 形態素境界 F1 | **0.8903** | 0.7389 | 0.5968 |
| 名詞+助詞 過結合率 | **0.25%** | 19.55% | 0.00% |
| 送り仮名保持率 | **55.10%** | 38.37% | 0.00% |
| 不正 UTF-8 学習トークン率 | **0.00%** | 2.17% | 0.65% |
| JA スループット | **73.6 MB/s** | 3.3 | 16.5 |
| 敵対的 1 MiB レイテンシ | **9.0 ms** | 391 | 79 |

BEG は過結合を 0.25% まで抑えながら送り仮名 55.1% を保持し、形態素境界 F1 は 0.8903（hf-bpe +15.1pt、hf-unigram +6.3pt）に達しました（なお tiktoken の過結合 0% は、日本語をほとんど結合せず送り仮名保持が 0% となる退化点です）。
クラス別 Phase-2 により EN 圧縮（4.340）は語彙 3〜6 倍の cl100k（3.945）と o200k（3.957）を上回り、JA 圧縮は助詞を分割する代償として hf-bpe との 1.7% 差（1.754 vs 1.784）にとどめ、不正 UTF-8 トークン 0%（hf-bpe は 2.17%）も 32k 語彙で維持しています。
なお表中の `cl100k*` は 100k–200k 語彙の外部参照であり、同語彙サイズ同士の直接比較は上表の `hf-bpe` 列（および全結果レポートの `hf-*` 行）を参照してください（参考として 4k/32k 合成データ評価の対照値も以下に示します）。

| 指標 | vocab 4k: asym-full / hf-bpe | vocab 32k: asym-full / hf-bpe |
| --- | --- | --- |
| JA 圧縮 (chars/token) | **4.608** / 4.343 | **5.696** / 5.529 |
| 不正 UTF-8 学習トークン率 | **0.00%** / 23.78% | **0.00%** / 3.55% |
| 送り仮名保持率 | **48.16%** / 20.00% | **48.16%** / 39.18% |
| 敵対的 1 MiB レイテンシ | **11.4 ms** / 552 | **15.2 ms** / 385 |

全結果レポートは [artifacts](eval/artifacts/report.md)（4k合成）・[artifacts32k](eval/artifacts32k/report.md)（32k合成）・[artifacts_real](eval/artifacts_real/report.md)（32k実データ）、可視化は [chart 4k](eval/artifacts/chart.html)・[chart 32k](eval/artifacts32k/chart.html)・[chart real](eval/artifacts_real/chart.html) に格納しています。

#### 200k モデルと外部トークナイザー横断比較

最終コーパス（jawiki 記事 440MB + 青空文庫 31.7MB + 既存 22.3MB［前節の ja.wikipedia + Gutenberg + OSS code］ + enwiki 記事 120MB + OSS code/docs ≈ 200MB ≈ 817MB）で学習した `asym-full`（語彙 200,000）は、同一レジスタの held-out jawiki 45.3M chars で 2.1287 chars/token を記録しました。
これは Sarashina3（API 専用・トークナイザー非公開）の公開実績値「日本語 Wikipedia 約19億文字 → 約9.5億トークン」= 2.0 chars/token と比べて +6.4% の chars/token に相当します（同一テキストならトークン数で約 -6.0%。ただし Sarashina3 の公開値は非公開テストセット上の数値であり、同一条件の直接比較ではありません）。

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

表中の `sarashina3 (公開値)` 列は非公開テストセット上の公表値のため同一条件比較ではなく、JA held-out 行の「勝者」は公開値を目標値として上回った参考比較です。実測同一条件での比較は sarashina2・o200k の各行を参照してください。
EN 圧縮でも `asym-full`（200k）は `tiktoken` o200k（3.9569）および cl100k（3.9448）を上回り、enwiki 無しの 709MB コーパスで学習した JA 特化モデル（holdout JA 2.1371・EN 3.814）と用途に応じて使い分けられます。
さらに pretokenizer の「改行+インデント結合」「ASCII 句読点の右向き結合」とコード増量コーパス（817MB）を用いた `kototok 200k (s3h)` について、同一テストセットで公開外部トークナイザー 10 モデルを実測し、各指標の最良値（外部側いずれか 1 モデルがとった最高値）と 8 軸で比較した結果を示します（境界指標は 150 文 hand gold 計測。JA 圧縮のみ、測定不能な sarashina3 の公開値を参考値として併記します）。

| 指標 | 最良の外部モデル | その値 | kototok 200k (s3h) | 勝者 |
| --- | --- | ---: | ---: | --- |
| JA 圧縮 held-out (chars/token) | sarashina3（公開値・別テストセット） | 2.0 | **2.179** | kototok（参考比較 +8.9%） |
| EN 圧縮 (chars/token) | tiktoken o200k | 3.957 | **4.356** | kototok (+10.1%) |
| Code 圧縮 (bytes/token) | llama3 / cl100k | 4.37 | **5.380** | kototok (+23.1%) |
| Mixed 圧縮 (bytes/token) | llama3 | 3.99 | **4.913** | kototok (+23.1%) |
| 形態素境界 F1 | plamo | 0.8749 | **0.9361** | kototok (+6.1pt) |
| 送り仮名保持率 | sarashina2 | 91.02% | **91.43%** | kototok |
| 名詞+助詞 過結合率 | o200k（非退化モデル中最良） | 0.75% | **0.25%** | kototok |
| エンコード速度 (JA) | tiktoken o200k | 14.6 MB/s | **~65 MB/s** | kototok (4.5x) |

過結合率では `cl100k` が 0.00% を記録しますが、これは日本語をほとんど結合しない退化点（送り仮名保持 0.00%・境界 F1 0.5968）で自明に達する値であり、結合を行うモデル間の比較対象からは外しています。

#### 多言語モデルとポリシー切替の圧縮比較（bytes/token 基準）

この節では `chars/token` ではなく、同一 held-out 入力に対する `bytes/token`（高いほど少トークン）で多言語モデル `v2-BKO3` とポリシー切替モデル `v2-T` を比較します。

| 入力 | v2-BKO3 | 外部最良 | 次点 kototok |
| --- | ---: | ---: | ---: |
| JA prose (364KB) | **5.79** | sarashina2 5.24 | s3n-BKO5 5.77 |
| Korean 未見 wiki (381KB) | **3.75** | llama3 3.58 / o200k 3.58 | s3n-BKO5 3.60 |
| Chinese 未見 wiki (101KB) | **4.01** | deepseek 3.77 | s3n-BKO5 3.65 |
| Casual/SNS edge | **3.97** | gemma2 3.72 | s3n-BKO5 3.81 |
| EN prose | **4.77** | o200k 4.11 | s3n-BKO5 4.57 |
| Code | **5.25** | llama3 / cl100k 4.37 | s3n-BKO5 5.05 |
| Mixed JA/code | **4.93** | o200k 3.99 | s3n-BKO5 4.74 |

同一モデル `v2-T`（`okuedge` 層追加前の構成）で `strict` と `compress` を切り替えた場合の held-out 実測を示します（KO/ZH/JA/EN/code/mixed 行はいずれも bytes/token 単位）。

| | strict | compress | 備考 |
|---|---:|---:|---|
| 境界 F1 / okurigana / 過結合 | **0.830 / 86.0% / 1.3%** | 0.821 / 74.6% / 1.1% | strict が歴代最良 F1・最少過結合 |
| KO 未見 | **3.75** | 2.88 | `kozh` レイヤーのみで +30% |
| ZH 未見 | **3.97** | 3.49 | 同上 |
| JA / EN / code / mixed | 5.69 / 4.77 / 5.25 / 4.93 | 5.63 / 4.77 / 5.25 / 4.91 | 差は微小 |

なお、この表は `okuedge` 層導入前の時点での比較であり、`okuedge` 層を含めた `kototok_v2s_200k.ktok` のプロファイル比較（`strict` 0.864 / 88.9% / 0.4% と `balanced` 0.865 / 86.0% / 1.4%）は次節の境界評価に掲載します。

#### 68 ドメイン多言語・数式・コード横断評価（264k）

公開データセット（`FineWeb-Edu` 英語、`FineWeb-2` 57 言語、`FineMath-4plus` 数式、`the-stack-smol-xl` 9 プログラミング言語）から学習用と完全に非重複の held-out 評価セット（計 68 ドメイン・11.16MB）を構築し、総トークン数（少ないほど高圧縮）を比較しました。
同一 286MB コーパスで学習した対照モデル `kototok-Global-Same` と `HF-Global-264k`、主要言語の配分を整えた完全非重複 1.03GB コーパスで学習した `kototok_global_264k.ktok`、および公開外部トークナイザー 11 モデル（`Custom Global 264,498` を含む計 12 外部モデル）の結果を示します（詳細 JSON は [eval/global_68/bench_final.json](eval/global_68/bench_final.json) と [eval/global_68/bench_same.json](eval/global_68/bench_same.json) に格納）。

| モデル | 語彙数 | Language (58言語) | Math | Code (9言語) | GrandTotal | `kototok-Global-264k` の勝数 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **kototok-Global-264k (`.ktok`)** | **264,498** | **1,652,373** | 192,024 | 417,166 | **2,261,563** | 全 12 外部モデルに **816/816**（仮想最良比 **68/68**） |
| **kototok-Global-Same（同一286MB）** | 264,498 | 1,678,138 | **174,634** | **381,685** | 2,234,457 | 同一条件対照（HF 比 -17.6%） |
| **kototok-Global-264k-HF（HF書出）** | 267,817 | 1,653,187 | 193,573 | 424,693 | 2,271,453 | 全 12 外部モデルに 811/816（7 モデルに 68/68） |
| Custom Global 264,498 (`SummerAI-O`) | 264,498 | 1,904,877 | 235,700 | 530,445 | 2,671,022 | **68/68**（GrandTotal -15.3%、速度 4.53x） |
| HF-Global-264k（同一286MB） | 264,498 | 1,915,986 | 235,507 | 561,297 | 2,712,790 | **68/68**（GrandTotal -16.6%） |
| GPT-4o (`o200k_base`) | 200,019 | 2,019,663 | 233,938 | 468,617 | 2,722,218 | **68/68**（GrandTotal -16.9%） |
| Gemma-3 (`gemma-3-1b-it`) | 262,145 | 1,952,061 | 252,891 | 564,176 | 2,769,128 | **68/68**（GrandTotal -18.3%） |
| Qwen3.5 (`Qwen3.5-0.8B`) | 248,070 | 2,056,155 | 253,266 | 512,632 | 2,822,053 | **68/68**（GrandTotal -19.9%） |
| Gemma-2 (`gemma-2`) | 256,000 | 2,090,868 | 250,432 | 562,829 | 2,904,129 | **68/68**（GrandTotal -22.1%） |
| DeepSeek-V3 | 128,815 | 2,340,419 | 231,462 | 503,145 | 3,075,026 | **68/68**（GrandTotal -26.5%） |
| Inkling-2 | 200,058 | 2,207,167 | 253,578 | 641,070 | 3,101,815 | **68/68**（GrandTotal -27.1%） |
| Qwen3 (`Qwen3-0.6B`) | 151,669 | 2,656,851 | 248,559 | 475,861 | 3,381,271 | **68/68**（GrandTotal -33.1%） |
| Qwen2.5 | 151,665 | 2,656,851 | 248,559 | 475,861 | 3,381,271 | **68/68**（GrandTotal -33.1%） |
| Llama-3 | 128,256 | 2,752,807 | 235,135 | 464,207 | 3,452,149 | **68/68**（GrandTotal -34.5%） |
| GPT-4 (`cl100k_base`) | 100,277 | 3,272,560 | 235,843 | 464,423 | 3,972,826 | **68/68**（GrandTotal -43.1%） |

- 同一 286MB コーパス条件の比較では、`kototok-Global-Same` が `HF-Global-264k` に対し GrandTotal で -17.6%（478,333 トークン削減）を達成しています。標準 ByteLevel BPE である `HF-Global-264k` や `Custom Global 264,498` は結合記号の分断によりインド系・東南アジア系言語でトークン数が増えるのに対し、kototok の多言語 DFA と階層フォールバック Phase-2 は言語・数式・コードの 3 領域すべてを同時に圧縮します。
- バランス配分コーパス（1.03GB）で学習した `kototok_global_264k.ktok` は、JA `119,725`（Custom Global `120,185`・o200k `187,516`）、ZH `158,261`（簡体字単独 `+1.13%`・繁体字単独 `+6.08%`、Custom Global `164,509`）、KO `86,868`（Custom Global `88,501`）、EN `210,582`（o200k `226,070`・Custom Global `229,307`）、Math `192,024`（DeepSeek-V3 `231,462`・Custom Global `235,700`）、Python `71,204`（Llama-3 `75,503`・Custom Global `86,514`）、ベンガル語 `9,352`（Gemma-3 `9,575`・Custom Global `27,372`）、タイ語 `11,161`（Qwen3.5 `11,361`・Custom Global `20,089`）を記録し、全 12 外部モデルに対して延べ 816/816 勝（各ドメインの外部最良値に対しても 68/68 全勝）を達成しています。さらに絵文字・国旗・ZWJ・結合記号・罫線・数式記号の Unicode ストレステストでも `1,055` トークン（HF 書出 `1,032`）となり、`Custom Global 264,498`（`1,597`）比で -33.9%（HF 版 -35.4%）に抑えています。
- Hugging Face 形式へ書き出した `kototok-Global-264k-HF` は、UTF-8 原子文字の中間バイト列を ByteLevel BPE 語彙として補完するため語彙数が 267,817（+3,319）になりますが、Language 合計は `.ktok` と 0.049% 差（814 トークン差）で一致し、全 68 ドメイン 11.16MB および Unicode ストレステストで完全可逆です（`Custom Global 264,498`・`HF-Global-264k`・`Gemma-3`・`Qwen3.5`・`Gemma-2`・`DeepSeek-V3`・`Inkling-2` の 7 モデルに 68/68、延べ 811/816 勝）。
- `kototok_global_264k.ktok` のエンコード速度は `eval/global_68/eval/jpn_Jpan.txt`（702KB）で 1 スレッド 56.24 MB/s（9.59M tokens/s）、8 スレッド 177.31 MB/s（30.24M tokens/s）であり、全 68 ドメイン 11.16MB 通し実行でも 15.17 MB/s（`Custom Global 264,498` 3.35 MB/s の 4.53 倍）です。

### 形態素境界と送り仮名保持

まず、150 行の hand gold セットで計測した境界品質（`kototok s3h` および外部 10 モデル）を示します。

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

外部モデルのうち `llama3` は名詞+助詞を 100% 過結合して日本語境界を保てておらず、`s3h` では `美|味しい` や `山手線` も 1 トークン化できました（ただし `気|持ちよく` は語彙競合により稀に分割が残ります）。
続いて、未見 500 文の Sudachi gold（500 unseen lines）における同一条件比較（Matched held-out comparison）を示します。

| tokenizer | boundary F1 | okurigana integrity | noun+particle overmerge |
|---|---:|---:|---:|
| **kototok s3o-BKO2** | **0.802** | **94.1%** | **4.9%** |
| sarashina2 | 0.783 | 86.2% | 26.6% |
| gemma2 | 0.767 | 41.8% | 18.8% |
| plamo | 0.690 | 72.8% | 27.4% |
| llm-jp / llama3 / qwen2.5 / deepseek | 0.21-0.22 | ~71% | 70-74% |
| mistral | 0.092 | 0.6% | 95.4% |

同じ Sudachi 500 文 holdout における `kototok_v2s_200k.ktok`（`okuedge` 層を含む）のプロファイル別指標を示します（JA 列は bytes/token 単位）。

| プロファイル（同一 `kototok_v2s_200k.ktok`） | F1 | oku | 過結合 | JA |
|---|---:|---:|---:|---:|
| strict（全タグ有効、`okuedge` 含む） | **0.864** | **88.9%** | **0.4%** | 5.12 |
| balanced（`--tags ja,casual,kozh,okuri`） | 0.865 | 86.0% | 1.4% | 5.39 |

このうち `strict` は名詞+助詞の過結合を 0.4% に抑えながら、先の `bytes/token` 比較表で次点に挙げた BKO 系比較モデル `s3n-BKO5`（87.3%）を上回る送り仮名保持 88.9% を達成し（送り仮名単体の数値では `s3o-BKO2` の 94.1% が最大）、`balanced` は JA 圧縮を温存する運用設定となります（なお `v2-BKO3` は再生成 Sudachi holdout 500 文で F1 0.823・送り仮名保持 82.9%・名詞+助詞過結合 1.5% でした）。
Sudachi の分割基準に依存した指標でないことを確かめるため、同じ 500 文を別系統辞書の UniDic（fugashi / unidic-lite、MeCab 短単位）で `eval/gold_unidic.py` により再分割し、`eval/boundary_holdout.py` で交差検証した結果を示します。

| tokenizer | boundary F1 | okurigana integrity | noun+particle overmerge |
|---|---:|---:|---:|
| **kototok v2-T strict** | **0.852** | **89.0%** | **0.5%** |
| kototok v2-T balanced | 0.850 | 86.0% | 1.5% |
| kototok s3o-BKO2 | 0.799 | 92.0% | 3.9% |
| gemma2 | 0.783 | 40.6% | 17.4% |
| sarashina2 | 0.770 | 83.5% | 26.0% |
| o200k | 0.762 | 2.2% | 0.3% |
| cl100k | 0.674 | 0.0% | 0.2% |
| plamo | 0.667 | 72.0% | 29.2% |
| llm-jp / llama3 / qwen2.5 / deepseek | 0.21-0.23 | ~72% | 69-73% |
| mistral | 0.088 | 0.4% | 94.8% |

独立 gold（UniDic）でも順位は保存されており、`kototok` の外科レイヤーが上位の境界 F1 を維持し、`v2-T strict` の送り仮名/過結合プロファイルは辞書系統を跨いでも安定しています（Sudachi 88.9% / 0.4% → UniDic 89.0% / 0.5%）。

### 下流言語モデル代理指標（trigram-LM）

トークナイザーごとに同一の 64MiB jawiki スライスで trigram 言語モデル（KN 継続カウント付き単一割引絶対バックオフ、`eval/downstream_trigram.py`）を学習し、363KiB の held-out jawiki テキストでバイト正規化交差エントロピー（bits/byte、低いほど良く圧縮と予測可能性を同時に測定）を計測しました。
kototok は上位 7 枠中 5 枠を占めてすべての汎用多言語トークナイザーを上回り、より大規模な日本語コーパスで学習された JA 特化の `sarashina2` とは 0.004 bits/byte 差に迫りました（一方で `v2-T strict` の境界ポリシーは助詞を分割する対価として、下表の他の BKO 系モデルである `s3o-BKO3 / s3o-BKO2`・`v2-BKO3`・`s3n-BKO5`・`v2-T balanced` より約 0.02 bits/byte コストが増えます）。

| tokenizer | bits/byte | eval tokens |
|---|---:|---:|
| sarashina2 | 1.433 | 69,587 |
| **kototok s3o-BKO3 / s3o-BKO2** | **1.437** | 64,885 |
| **kototok v2-BKO3** | **1.441** | 62,930 |
| **kototok s3n-BKO5** | **1.443** | 63,060 |
| plamo | 1.447 | 73,397 |
| **kototok v2-T balanced** | **1.448** | 67,577 |
| **kototok v2-T strict** | **1.464** | 71,334 |
| gemma2 | 1.485 | 82,440 |
| llm-jp | 1.498 | 89,950 |
| deepseek | 1.534 | 94,105 |
| llama3 | 1.557 | 99,734 |
| qwen2.5 | 1.560 | 100,819 |
| o200k | 1.590 | 108,183 |
| mistral | 1.730 | 147,947 (2.3x kototok) |
| cl100k | 1.763 | 149,199 |

## 配布モデル

学習済みモデル 6 種（`.ktok` バイナリ）と、Rust のビルドなしでそのまま読み込めるフラッグシップの Hugging Face 形式ディレクトリ（`models/kototok_global_264k_hf/`）を `models/` に同梱しています。
`kototok` CLI・Python バインディングは `.ktok` / `json` / `hf` / `hf-dir` / `tiktoken` / `bpe` の全 6 形式を自動で読み書きでき、他のモデルも `kototok export` で任意の形式へ変換できます。

```python
# Rust ビルド不要（Hugging Face tokenizers / transformers でそのまま利用可能）
from tokenizers import Tokenizer
tok = Tokenizer.from_file("models/kototok_global_264k_hf/tokenizer.json")
# または transformers: AutoTokenizer.from_pretrained("models/kototok_global_264k_hf")
```

| ファイル | サイズ | 内容 |
| --- | ---: | --- |
| `models/kototok_global_264k_hf/` | 14.6MB | `kototok_global_264k` の Hugging Face 形式ディレクトリ（`tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json`、語彙 267,817）。`transformers`・`tokenizers`・vLLM 等からビルド不要で直接読込可能（68 ドメイン完全可逆、`Custom Global 264,498` に 68/68・Unicode ストレステスト -35.4%） |
| `models/kototok_global_264k.ktok` | 39.6MB | 68 ドメイン（58 言語＋数式＋コード 9 言語、1.03GB バランスコーパス＋Unicode 17 ブロックシード）で `--preset global` 学習した多言語フラッグシップモデル。alive vocab 264,498、`invalid_utf8_learned_tokens = 0`、68 ドメイン評価で `Custom Global 264,498` を含む外部 12 モデルに 816/816（仮想最良比 68/68）全勝 |
| `models/kototok_v2s_200k.ktok` | 44.7MB | v2 ベース（776MB コーパス・vocab 200k）に複数 refine を適用したポリシー切替モデル。`tag_ranges`（ja/casual/kozh/okuri/okuedge）+ `tagged_prunes`（okuedge 3,482 件）を内蔵し、alive vocab 281,630。`--mode strict`（既定）/ `--tags ja,casual,kozh,okuri`（balanced）を切替可能 |
| `models/kototok_v2_bko3_200k.ktok` | 41.6MB | 同 v2 ベースに JA/口語の Sudachi gold と KO/ZH の whitespace・jieba gold で Aligned BPE-Knockout（1,328 prune + 64,498 graft、モデル内メタデータ記録の適用 gold は 17,536 文）を施した多言語モデル。alive vocab 264,498 |
| `models/kototok_s3o_bko2_200k.ktok` | 34.2MB | s3 系ベース（817MB コーパス・vocab 200k）に外科的 BKO2 を適用したモデル（Sudachi holdout 送り仮名保持 94.1%、trigram-LM 1.437 bits/byte） |
| `models/kototok_s3o_bko3_200k.ktok` | 34.2MB | 同 s3 系ベースに外科的 BKO3 を適用したモデル（trigram-LM 1.437 bits/byte） |
| `models/kototok_s3n_bko5_200k.ktok` | 38.0MB | 同 s3 系ベースに外科的 BKO5 を適用したモデル（JA prose 5.77 bytes/token、trigram-LM 1.443 bits/byte） |

なお `kototok_global_264k.ktok` は事後 graft を施していないため `invalid_utf8_learned_tokens = 0` です。v2-T と v2-BKO3 は学習時の UTF-8 原子性によりベースモデルの不正 UTF-8 学習トークンは 0 ですが、事後修復の graft 層が文字境界をまたぐバイト断片を中間ピースとして追加するため、`inspect` 上の `invalid_utf8_learned_tokens` は v2-T が 1,004 件（0.36%）、v2-BKO3 が 1,069 件（0.40%）となります。これらはエンコード時にそのまま出力され得る語彙内断片であり、decode はバイト列連接で復元するため完全可逆性には影響しません（全配布モデルで往復が完全に一致することを確認済み）。

出所: `kototok_global_264k.ktok` は本リポジトリの `kototok train --preset global` で学習したもので、学習コーパスは Hugging Face 上の公開データセット（FineWeb-Edu・FineWeb-2・FineMath-4plus・the-stack-smol-xl）と ja.wikipedia に由来します。v2 系・s3 系はいずれも本リポジトリの `kototok train`（`asym-full` ベース）で学習後、`kototok refine` で gold 境界修復を適用したものであり、学習コーパスは ja.wikipedia（CC BY-SA）・enwiki・青空文庫・OSS コード、境界修復の gold は Sudachi / UniDic 系の分割に由来します。
ライセンス: モデルファイル自体は本リポジトリと同一の MIT OR Apache-2.0 デュアルライセンスです。学習データ由来の制約（jawiki の CC BY-SA 等）は利用目的に応じて利用者側で確認してください。

整合性チェックは同梱の `models/SHA256SUMS.txt` で検証できます。

```bash
(cd models && sha256sum -c SHA256SUMS.txt)
```

## 再現手順

### ワークスペースと CLI 基本操作

| クレート | 役割 |
| --- | --- |
| `crates/kototok-core` | DFA 前処理・エンコード/デコード・6 形式入出力・モデル定義（60 テスト） |
| `crates/kototok-train` | Rayon 並列トレーナー（Phase-1 BPE + 階層フォールバック Phase-2 + Unicode シンボルシード、21 テスト） |
| `crates/kototok-cli` | `kototok` バイナリ（`train` / `encode` / `decode` / `export` / `convert` / `inspect` / `viz` / `refine` / `bench`） |
| `crates/kototok-py` | PyO3 バインディング（`import kototok` で全 6 形式の load/save/encode/decode/train） |

```bash
cargo build --release          # バイナリ: target/release/kototok
cargo test --workspace         # 全テスト（84 件・7 スイート）
cargo clippy --workspace --all-targets -- -D warnings

# 学習 → 検査 → エンコード（出力ファイル名や --format で 6 形式を直接読み書き）
kototok train  --input corpus.txt --output model.ktok --vocab-size 4096 --preset asym-full
kototok train  --input corpus.txt --output tokenizer.json --vocab-size 4096 --preset global
kototok inspect --model model.ktok --json
echo "日本語のテキスト" | kototok encode --model model.ktok --input /dev/stdin --json
kototok bench  --model model.ktok --input corpus.txt --threads 1 --iters 5 --json

# 汎用フォーマット変換（ktok / json / hf / hf-dir / tiktoken / bpe）
kototok export --model models/kototok_global_264k.ktok --output tokenizer.json
kototok export --model models/kototok_global_264k.ktok --output hf_out --format hf-dir
kototok export --model models/kototok_global_264k.ktok --output global.tiktoken
kototok export --model models/kototok_global_264k.ktok --output bpe_out --format bpe
kototok encode --model tokenizer.json --text "新しいトークナイザー" --pieces
```

語彙サイズ 200,000（817MB コーパス）での学習レシピは次のとおりです。

```
kototok train --input eval/artifacts_real/data/train_corpus_s3f.txt \
  --vocab-size 200000 --phase2-min-npmi 0.30 \
  --phase2-min-npmi-okuri 0.0 --beg-okuri-boost 20
```

学習コーパス `train_corpus_s3f.txt`（約 780MB）はリポジトリに同梱せず、次のコンポーネントをこの順に連結して構築しました。

1. `eval/realdata_builder.py` が生成する `artifacts_real/data/train_corpus.txt`（22.3MB: ja.wikipedia + Gutenberg + OSS code）
2. jawiki の pages-articles ダンプを `eval/jawiki_extract.py <dump.bz2> train_corpus_jw.txt` で平文化したもの（約 400MB 規模の記事テキスト）
3. `eval/aozora_fetch.py --mb 30 --out aozora.txt`（青空文庫 約 31.7MB）
4. `eval/enwiki_fetch.py --mb 120 --output enwiki.txt`（enwiki 記事 約 120MB）
5. OSS のソースコード・ドキュメント類の結合テキスト（約 200MB。ライセンス上再配布可能なコードから任意に選定）

`cat` で上記順に連結し、`eval/artifacts_real/data/train_corpus_s3f.txt` として配置すれば同一レシピで学習できます。なお 2〜5 の分量や語種比率で結果は変動するため、正確な再現には各成分の実測バイト数をそろえてください（README の評価節に記載の内訳を参照）。

このレシピで学習したモデル（vocab 200,000）におけるエンコード例は次のとおりです。

```
日本|の|首都|で|食べる             # 名詞+助詞は分割、活用形は1トークン
書き換える / 読み取る / 使いやすく   # 連用形複合・形容詞形も1トークン
美味しい / 気持ちよく / 山手線        # 従来割れていた語も1トークン
方法|を|試す                       # 助詞分割 + 動詞保持
```

### Python バインディングとポリシー切替

```bash
PYO3_PYTHON=$(which python3) cargo build --release -p kototok-py
cp target/release/libkototok.so pylib/kototok.so   # sys.path 上に置く
# または maturin: cd crates/kototok-py && maturin develop --release
# または pip（ABI タグ付きで目的の環境へインストール）: pip install ./crates/kototok-py
```

拡張モジュールはビルド時の CPython バージョンに依存します（PyO3 はバージョン別バイナリを生成します）。インタプリタを更新・変更した場合（例: Python 3.13 用にビルドした `.so` を 3.14 で読み込む）はインポート失敗やクラッシュの原因になるため、他環境でビルド済みの `.so` は使い回さず、必ず利用中のインタプリタで再ビルドしてください。リポジトリにはビルド済み `.so` を同梱していないため、fresh clone では上記のビルド手順が必須です。

```python
import kototok

tok = kototok.train("corpus.txt", output="model.ktok", vocab_size=4096, preset="asym-full")
# .ktok / tokenizer.json / hf-dir / .tiktoken / bpe-dir / .json を自動判定して読込可能
tok = kototok.Tokenizer.load("models/kototok_global_264k.ktok")
print(kototok.detect_format("models/kototok_global_264k.ktok"), kototok.formats())
tok.save("tokenizer.json")          # 拡張子・ファイル名から自動推定（format="hf-dir" 等で明示も可）
tok.save_tiktoken("global.tiktoken")

ids    = tok.encode("新しいトークナイザーの設計を調べる。")
pieces = tok.pieces("新しいトークナイザーの設計を調べる。")
text   = tok.decode(ids)            # 完全可逆ラウンドトリップ
batch  = tok.encode_batch(["東京で会議が始まる。", "hello world"])
```

ポリシー切替用の外科層を付与するには、`refine --tag ja|casual|kozh|okuri` や `refine --prune-oku-edges`（`--tag okuedge`）を実行します。
推論時は CLI の `encode --mode strict` / `--mode compress` / `--tags kozh,okuri`（balanced 構成では `--tags ja,casual,kozh,okuri`）、Python では `Tokenizer.load(path, tags=[...])` で有効化する層を指定します。

### ベンチマークと境界・下流評価の再現

合成データ（4k/32k）および実データ（32k）でのベンチマーク（10 モデル）は次の手順で再現できます。

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

500 文 holdout の境界指標と下流代理指標は、`pylib/kototok.so` と `eval` 依存パッケージを用意したうえで、`eval/gold_unidic.py`（UniDic gold 生成）、`eval/boundary_holdout.py`（Sudachi/UniDic 境界評価）、および `eval/downstream_trigram.py`（64MiB jawiki 学習・363KiB held-out 評価の trigram-LM 交差エントロピー計測）で再現できます。

68 ドメイン多言語・数式・コード評価および `kototok_global_264k.ktok` の学習は次の手順で再現できます。

```bash
# 68 ドメインの非重複評価セット（11.16MB）と学習コーパスを構築
eval/.venv/bin/python eval/global_corpus_builder.py --workers 8            # 286MB 同一コーパス対照用
eval/.venv/bin/python eval/global_corpus_builder.py --balanced --workers 8 # 1.03GB バランス学習用

# フラッグシップ学習（--preset global、vocab 264,498）
./target/release/kototok train --preset global --vocab-size 264498 --threads 4 \
  --input eval/global_68/train_global_68_balanced.txt --output models/kototok_global_264k.ktok

# 全 68 ドメイン比較ベンチマーク
eval/.venv/bin/python eval/global_bench.py \
  --model "GPT-4o (o200k)=tiktoken:o200k_base" \
  --model "GPT-4 (cl100k)=tiktoken:cl100k_base" \
  --model "HF-Global-264k=hf:eval/global_68/hf_global_264k.json" \
  --model "kototok-Global-264k=ktok:models/kototok_global_264k.ktok" \
  --model "kototok-Global-264k-HF=hf:models/kototok_global_264k_hf/tokenizer.json"
```

## 制約と今後の課題

- Sudachi 境界指標は「Sudachi への一致率」を測る自己言及的評価であり（UniDic 交差検証で順位保存を確認済み）、ベースモデル自体は辞書非依存である一方、Aligned BPE-Knockout の外科的修復は gold 分割が得られない言語にはそのまま適用できません。
- Sarashina3 比較用テストセットの非公開性と Casual/SNS 合成エッジセットの少数性（数行規模）により外部比較カバレッジには限界があり、`v2-BKO3` の境界数値も再生成した別 holdout に基づくため同一条件の口語・大規模評価が不足しています。

運用時のモデル選択では、多言語圧縮・送り仮名保持・下流言語モデルの予測コストの間にトレードオフが存在し、検証範囲にも今後の課題が残っています。

- 多言語混合モデル（`v2-BKO3`）では KO/ZH 混入により日本語ネイティブの merge が薄まる構造的コストがあるため、送り仮名保持は `s3n-BKO5`（87.3%）のほうが高く（`v2-BKO3` は 82.9%）、送り仮名を最重視する用途では `s3n-BKO5`（または `v2-T` strict の 88.9%）を選びます。
- 今後の課題として、trigram-LM 代理指標にとどまらないニューラル言語モデルでの下流検証（`eval/downstream_lm.py` で小規模 GPT の実訓練比較を準備中・未検証）および Casual/SNS 領域の評価セット拡充が残っています。

## ライセンス

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) のデュアルライセンスです。
