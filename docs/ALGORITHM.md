# AsymScript-BPE 形式仕様 (`kototok`)

本書は、日本語・英語・ソースコードが混在する入力向けのサブワード・トークナイザー **AsymScript-BPE** (`kototok`) の形式仕様、不変条件、計算量証明、および評価手順を定める。

---

## 1. 背景と日本語トークナイゼーションの4つの病理

従来の Byte-level BPE (BBPE) や英語中心の前処理規則を日本語へそのまま適用すると、次の4つの構造的問題が生じる。

| 問題 | 原因 | 発生する現象 |
| --- | --- | --- |
| **1. 3バイト UTF-8 断片化** | `0..256` の生バイトを無制約にマージする (`RawBytes`) | 日本語のひらがな・カタカナ・漢字は UTF-8 で主に 3 バイト（補助漢字や絵文字は 4 バイト）をとる。先頭バイトや継続バイト (`0x80..0xBF`) 単体の断片が語彙を占め、不正な UTF-8 バイト列を含む学習済みトークンが増える。 |
| **2. 空白の欠如と送り仮名・助詞ジレンマ** | 文字種遷移を無視して結合するか、一律に分割するかの二択 | `StandardGpt` のように全 Unicode 文字をまとめると「名詞＋助詞」（例: `日本の`, `開発者が`）が 1 トークンへ過結合する。逆に `StrictScript` のように文字種境界をすべて切ると、「漢字語幹＋送り仮名」（例: `食べる`, `新しい`）が `食 \| べる` のように分断される。 |
| **3. PickyBPE 以前の中間トークン死蔵** | マージの途中段階で作られた部分語が語彙に残る | ペア $(A, B) \to AB$ のマージ後に $A$ や $B$ が高次トークンへほぼ完全に吸収されても語彙枠を占有し続け、推論時に一度も出現しない死蔵トークン (Dead Token) が増加する。 |
| **4. 正規表現バックトラックの $O(N^2)$ 罠** | 前処理の事前分割にバックトラック付き正規表現を使う | 同一文字や特定パターンの長い連続入力（例: `あ` が 1MB 連続する文字列）に対して、先読み・後読みや貪欲量化子の巻き戻しが起き、最悪 $O(N^2)$ 時間で遅延する。 |

AsymScript-BPE は、これらを 3 つの柱（非対称文字種遷移 DFA、UTF-8 原子性不変条件と ACSG・PickyBPE による二段階語彙学習、最悪 $O(N)$ かつ完全可逆なエンコード）で同時に解消する。

---

## 2. 第1の柱: 非対称文字種遷移 DFA (`PreTokenizer`)

### 2.1 文字クラス定義

`crates/kototok-core/src/model.rs` の分類規則に準拠し、入力の Unicode スカラー値 $c$ を次の文字クラスへ写像する。

| 文字クラス | Unicode 範囲・条件 |
| --- | --- |
| `Kanji` | `U+4E00..=U+9FFF`, `U+3400..=U+4DBF`, `U+F900..=U+FAFF`, `U+20000..=U+2FA1F`, `U+3005..=U+3007` |
| `Hiragana` | `U+3041..=U+3096`, `U+309D..=U+309F`（長音符 `U+30FC` は直前の仮名種別を継承） |
| `Katakana` | `U+30A1..=U+30FA`, `U+30FD..=U+30FF`, `U+FF66..=U+FF9F`, 単独時または直前がカタカナの `U+30FC` |
| `AsciiWord` | ASCII 英字 `[A-Za-z]` および `_` |
| `Digit` | ASCII 数字 `[0-9]` および Unicode 10 進数字 |
| `Space` | 半角スペース `U+0020` |
| `Newline` | `\n`, `\r` |
| `Punct` | 句読点・括弧・記号類（ASCII 記号、和文句読点 `、。・「」『』（）` 等） |
| `Other` | 上記以外の Unicode 文字（絵文字、非 CJK 多言語文字等） |

### 2.2 3つの前処理モード (`PreTokenizerMode`)

`PreTokenizerConfig` の `mode` により、文字遷移 $(q_{\text{prev}} \to q_{\text{curr}})$ におけるチャンク継続・分割規則を切り替える。

| モード | `Kanji -> Kanji` | `Kanji -> Hiragana` | `Hiragana -> Kanji` | `Hiragana -> Hiragana` | `Katakana -> Katakana` | `Ascii <-> CJK` |
| --- | --- | --- | --- | --- | --- | --- |
| **`AsymScript`** | 継続 | **継続**（送り仮名許容） | **分割**（助詞と後続漢字を分離） | 継続 | 継続 | 分割 |
| **`StrictScript`** | 継続 | **分割** | **分割** | 継続 | 継続 | 分割 |
| **`StandardGpt`** | 継続 | 継続 | 継続 | 継続 | 継続 | 分割（全 Unicode レターを一体化） |

`AsymScript` の非対称性（`Kanji -> Hiragana` は許し、`Hiragana -> Kanji` は切る）により、`食べる` や `新しい` は 1 チャンク内に留まり、`食べた後に会議へ` の `に会議` は `に | 会議` で必ず分割される。

### 2.3 共通ガード条件とハード・ソフト境界

`PreTokenizerConfig` の各パラメータにより、すべてのモードで以下を制御する（実装は `crates/kototok-core/src/pretok.rs`）。

1. **ASCII 先頭空白結合 (`attach_ascii_leading_space = true`)**: 直前の半角スペース 1 文字 (`' '`) のあとに `AsciiIdent`・`AsciiDigit`・`AsciiOperator` が続く場合、空白を次チャンクの先頭に結合する（例: `" hello"`, `" 123"`, `" ="`）。2 文字以上の空白連や非 ASCII 直前の空白は結合対象外で、空白チャンクとして最大 8 文字 (`MAX_SPACE_RUN`) ごとに切り出される。
2. **数字列上限 (`max_digit_run = 3`)**: 連続する `Digit` が 3 桁に達した時点でチャンクを切る（例: `20261122` $\to$ `202 | 611 | 22`）。先頭空白結合された空白は桁数に算入しない。
3. **ASCII 演算子連上限 (`MAX_OPERATOR_RUN = 3`)**: `AsciiOperator` は最大 3 文字連でひとつのチャンクとする（例: `====` $\to$ `=== | =`）。
4. **最大チャンク長ガード (`max_chunk_chars = 128`)**: 同一チャンク内の文字数が `max_chunk_chars` に達した場合、文字種遷移にかかわらず強制分割する。これにより後続の局所 BPE 探索が定数サイズ以下に抑えられる。
5. **最大セグメントチャンク数 (`max_segment_chunks = 64`)**: Phase 2 における隣接チャンク結合の単位であるセグメントは、最大 64 チャンクで区切る。学習側のペア統計は `Vec<u32>` キーの肥大を防ぐため 16 チャンクにキャップして数える（エンコード時の上限 64 は変わらない）。

各 `PreToken` が持つ `hard_break_after` フラグがチャンク右縁のハード・ソフトを決める。チャンク間の境界は次の 2 種類に分類する。

- **ハード境界 (Hard Boundary, `hard_break_after = true`)**: 次のいずれかの地点。ハード境界は Phase 1 でも Phase 2 でも跨がない。
  - **改行トークンの右縁（条件付き）**: `\r\n`/`\n`/`\r` は単独トークンとなり、直後が空白・タブ・さらなる改行で**ない**場合のみハード。直後がインデントや空行の場合はソフトとなり、コードで頻出する「改行＋インデント」や空行連を右向きに結合できる（`newline+indent` merge）。一方、`改行＋単語` は必ずハードで、日本語の文境界は保たれる。
  - **空白チャンクの右縁**: 先頭空白結合に乗らなかった空白ラン（複数空白・タブ・非 ASCII 前置の空白）はハードで区切る。
  - **和文句読点類の両側**: `JaPunct`（`、。・「」『』（）` 等）および語頭の長音符 `ー` は、自身の右縁がハードであるうえ、**直前トークンの右縁も封印**する（例: `作る|`・`|`。|次`）。
  - **ASCII 句読点の直前（条件付き）**: `AsciiPunctHard` の直前が英数字・`_`・非 ASCII バイトで終わるトークン（`word` や漢字など）の場合、その右縁を封印する（`word,` は `word | ,` に割れる）。句読点・空白直後など非単語直前は封印しない。
  - **`max_segment_chunks` 超過地点**: セグメントが 64 チャンクに達した時点で強制フラッシュする。
- **ソフト境界 (Soft Boundary, `hard_break_after = false`)**: 上記以外のすべてのチャンク間境界。Phase 1 では越えないが、Phase 2 の Superword マージでは結合対象となる。具体的には次を含む。
  - **スクリプト遷移境界**: `Hiragana -> Kanji`（`に | 会議`）、CJK と ASCII の境界など、DFA が分割するが `hard_break_after` は立たない境界。`名詞+助詞` の誤結合防止はハード境界ではなく ACSG/BEG ゲート（§3.5）の責務であり、複合語・隣接チャンク結合は Phase 2 がソフト境界越しに学習する。
  - **ASCII 句読点トークンの右縁**: `,`・`(`・`'` などは右向きに結合でき、`():`・`, the`・`(self`・`'t` のようなコード・英語頻出形を作れる。
  - **改行＋インデント／空行**: 上記の条件付きソフト化した改行トークン。
  - 数字チャンク・演算子チャンク・空白結合済み ASCII チャンク・各種文字種ランの右縁。

---

## 3. 第2の柱: UTF-8 原子性・ACSG・PickyBPE・Phase 2

### 3.1 UTF-8 原子性不変条件 (`ByteMode::Utf8Atomic`)

`InternalToken` の内部 ID 空間（`internal_id: u32`）を次のように構成する。

| 内部 ID 範囲 | `TokenKind` | 内容とマージ可否 |
| --- | --- | --- |
| `0 .. 128` | `Ascii` | 1 バイト ASCII 文字（`0x00..0x7F`）。マージ可能。 |
| `128 .. 256` | `ByteFallback` | 非 ASCII 生バイト（`0x80..0xFF`）。**マージ不可 (non-mergeable)**。 |
| `256 ..` | `Special` | `<\|endoftext\|>` 等の特殊トークン。マージ不可。 |
| `256 ..` | `AtomicChar` | 学習コーパスに出現した有効な非 ASCII 1 文字（UTF-8 の 2〜4 バイト）。マージ可能。 |
| `256 ..` | `MergedPhase1` | Phase 1 チャンク内マージで生成されたトークン。 |
| `256 ..` | `MergedPhase2` | Phase 2 セグメント内マージで生成されたトークン。 |

対照ベースライン用の `ByteMode::RawBytes` では `0..256` の全生バイトをマージ可能とする。

**定理 (UTF-8 完全性定理 / Zero Invalid UTF-8 Learned Tokens)**
`ByteMode::Utf8Atomic` のもとで学習されたモデルにおいて、`audit_utf8_validity()` が報告する `invalid_utf8_learned_tokens` は常に $0$（`invalid_utf8_rate == 0.0`）である。

**証明**:
1. **基底**: マージ対象となる初期トークンは `Ascii`（1 バイト `0x00..0x7F`、有効な UTF-8）および `AtomicChar`（Unicode スカラー値の有効な UTF-8 エンコード列）のみである。`ByteFallback`（`128..256`）はマージ候補から除外される。
2. **帰納ステップ**: 任意の `VocabEvent::Merge { left, right, result }` は、マージ可能なトークン $l, r$ のバイト列の連接 $\text{bytes}(m) = \text{bytes}(l) \parallel \text{bytes}(r)$ として `MergedPhase1` または `MergedPhase2` を生成する。UTF-8 は自己同期・接頭辞自由な符号化であるため、2 つの有効な UTF-8 バイト列の連接は常に有効な UTF-8 バイト列である。
3. **結論**: 帰納法により、すべての `AtomicChar`, `MergedPhase1`, `MergedPhase2` トークンは有効な UTF-8 バイト列であり、`std::str::from_utf8(&tok.bytes)` は必ず成功する。一方で未知文字や不正バイト入力に対しても `ByteFallback` が維持されているため、エンコードの可逆性は損なわれない。 $\square$

### 3.2 ACSG (Asymmetric Cross-Script Cohesiveness Gating)

`AsymScript` DFA は `Kanji -> Hiragana` を同一チャンク内に残すため、送り仮名（例: `食` + `べる`）を学習できる反面、高頻度な「漢字名詞＋助詞」（例: `日本` + `の`, `開発` + `が`）も同一チャンクに現れる。これを頻度のみでマージすると助詞の過結合が起きる。

ACSG は、候補ペア $(l, r)$ のうち **$l$ が漢字を含み（または末尾が漢字であり）かつ $r$ の先頭がひらがなである `Kanji -> Hiragana` 跨ぎペア** に限り、正規化自己相互情報量 (NPMI) による凝集度ゲートを課す。

コーパス上の現在の総トークン出現数を $T$、$l, r$ の単独出現頻度を $f(l), f(r)$、隣接ペア頻度を $f(l, r)$ とおくとき、PMI および NPMI は次で定義される。

$$\text{PMI}(l, r) = \log \frac{f(l, r) \cdot T}{f(l) \cdot f(r)}, \qquad \text{NPMI}(l, r) = \frac{\text{PMI}(l, r)}{-\log \frac{f(l, r)}{T}} \in [-1, 1]$$

ACSG の有効スコア $S_{\text{ACSG}}(l, r)$（べき乗係数 $\beta > 0$）を次式で与える。

$$S_{\text{ACSG}}(l, r) = \begin{cases} f(l, r) \cdot \max\bigl(0,\, \text{NPMI}(l, r)\bigr)^\beta & (\text{cross-script } \text{Kanji} \to \text{Hiragana} \text{ の場合}) \\ f(l, r) & (\text{それ以外の場合}) \end{cases}$$

`食` と `べる` のように共起が相互に強く結びつくペアは $\text{NPMI}$ が高くスコアが減衰しにくい。一方、`日本` と `の` のように `の` が多数の名詞と共起して $f(\text{の})$ が極めて大きいペアは $\text{NPMI}$ が低くなり、$\beta$ 乗ペナルティによりマージ順位が下がる。

### 3.3 PickyBPE IoS による語彙精緻化と時系列イベント再生の単調性

標準 BPE ではマージ $(l, r) \to m$ の実行後も $l, r$ が無条件で語彙に残る。`kototok` では PickyBPE に基づく IoS (Intersection-over-Self) 剪定を導入する。

マージ $(l, r) \to m$ により $f_{\text{merged}}(m)$ 回の結合が起き、マージ後の残余単独頻度が $f_{\text{after}}(x)$ になったとき、構成要素 $x \in \{l, r\}$（ただし `Ascii`, `AtomicChar`, `ByteFallback`, `Special` などの基本トークンを除く学習済みトークン）の吸収率 IoS を次式で定義する。

$$\text{IoS}(x \mid m) = \frac{f_{\text{merged}}(m)}{f_{\text{after}}(x) + f_{\text{merged}}(m)}$$

$\text{IoS}(x \mid m) \ge \tau_{\text{IoS}}$（かつ $x$ が基本原子トークンでない）場合、$x$ は中間トークンとして実質的に $m$ へ吸収されたとみなし、語彙から剪定する。剪定時は `x.final_id = None` とし、その空きスロットを後続の有用なマージへ再利用する。

語彙変更履歴は単なる静的マージ表ではなく、時系列配列 `events: Vec<VocabEvent>` として記録する。

- `VocabEvent::Merge { left, right, result }`: 時刻（イベント添字）$t$ で隣接ペア `(left, right)` を `result` へ結合する。
- `VocabEvent::Prune { target, split_into }`: 時刻 $t$ でトークン `target` が残存していれば、その構成要素列 `split_into`（`target` を生んだ直前の有効子トークン列を再帰展開したもの）へ分割し戻す。

**単調性 (Chronological Event-Rank Replay Monotonicity)**:
エンコード時、入力チャンク内の初期トークン列に対し、`events` の添字（イベント・ランク $t = 0, 1, \dots, |E|-1$）の**厳密な昇順**で適用可能な最小ランクのイベントを逐次再生する。
1. 各 `Prune { target, split_into }` は、`target` を生成した `Merge` ランクより必ず大きいランク $t_{\text{prune}} > t_{\text{create}}$ に配置される。
2. また `split_into` に含まれるトークンは、`target` 剪定時点で既に生存保証された（もしくはさらに上位の `Prune` で展開可能な）要素である。
3. したがって、時刻 $t = |E|-1$ まで単調にイベントを再生し終えたトークン列には、`final_id == None` の剪定済みトークンが一切残らない。これにより、エンコード結果に剪定済み ID が混入する `KototokError::PrunedTokenEmitted` が構造的に回避される。

### 3.4 Phase 2: ハード・ソフト境界制御付き Superword マージ

`ModelMetadata` は `phase1_vocab_size` と `phase2_start_rank` を保持する。語彙学習は次の 2 フェーズで行う（`asym-full` 既定では `phase1_vocab_ratio = 0.75`）。

1. **Phase 1 (`MergedPhase1`)**: チャンク内部のみで BPE + ACSG/BEG + PickyBPE を実行し、目標語彙数の一部（例: `phase1_vocab_size`、全学習枠の大部分）まで語彙を構築する。
2. **Phase 2 (`MergedPhase2`)**: `phase2_start_rank` 以降の残余語彙枠を使い、**ハード境界を越えないセグメント（最大 `max_segment_chunks = 64` チャンク）内**で、ソフト境界を跨ぐ隣接チャンク代表列・複合トークン間の高頻度マージを学習する。

ハード境界（§2.3 の `hard_break_after` 地点: 条件付き改行・空白ラン・和文句読点両側・`word` 直後の ASCII 句読点前・セグメント長上限）でセグメントを切るため、日本語の文境界や句読点越えの誤結合を起こさずに、複合名詞や英語の頻出語連鎖を `MergedPhase2` として圧縮できる。スクリプト遷移境界はソフトであり、`に | 会議` 型の結合可否は Phase 2 の NPMI 床と BEG 右ゲートが制御する。

Phase 2 の候補採用にはクラス別 NPMI 床と ASCII クォータを課す（`phase2_npmi_floor`・`phase2_ascii_quota_reached`）。

- **純 ASCII ペア**: `phase2_min_npmi_ascii`（`asym-full` で 0.10）。日本語形態素リスクを持たないため緩い床で英語 superword を獲得する。
- **`Kanji -> Hiragana` ペア**: 右側ひらがな列の `gate_R < 0.9` なら `phase2_min_npmi_okuri`（活用形として低床化）、`gate_R >= 0.95` は `beg_hard_veto` で不採用、中間帯は通常床 `phase2_min_npmi` を維持する。
- **送り仮名尾ペア (`確か|める` 型)**: 左辺が漢字+かな語形・右辺が全ひらがなのペアは、結合後の連続かな列を `gate_R` テーブルで照合し、`< 0.9` なら `phase2_min_npmi_okuri` とする。
- **連用形複合 (`書き|換える` 型)**: 左辺が漢字を含みひらがな終わり・右辺が漢字頭かつひらがなを含む動詞形のペアも `phase2_min_npmi_okuri` とする。
- **ASCII クォータ**: 純 ASCII マージは Phase 2 予算の `phase2_ascii_quota`（`asym-full` で 30%）までに制限し、日本語 superword 枠を保護する。

Phase 2 のスコアは `f * (1 + NPMI)` を基底とし、BEG が有効 (`beg_strength > 0` かつ `beg_phase2`) な `Kanji -> Hiragana` ペアには右ゲートのみで `NPMI^(beg_strength * g_r)` のペナルティを掛ける。Phase 1 で割れた `食|べる` を Phase 2 superword が再接合する救済経路を残すため、Phase 2 では左ゲート `g_l` を使わない。`beg_okuri_boost` は `score * (1 + boost * (1 - g_r))` を低ゲート右辺と連用形複合に適用する（連用形複合は `1 + boost` のフルブースト）。

### 3.5 BEG (Boundary-Entropy Gate): 双方向エントロピーゲート

ACSG の一様指数 $\beta$ では `食 + べる`（語幹＋送り仮名）と `日本 + の`（名詞＋助詞）を区別できない。BEG は両者をコーパス統計のみで分離するため、`Kanji -> Hiragana` ペアに左右 2 つのゲートを適用する（実装は `crates/kototok-train/src/beg.rs`、学習時に `k2h_counts` として `Kanji -> Hiragana` チャンク内の `(漢字プレフィックス, ひらがな連)` 出現を集計する）。

1. **左ゲート `g_l`（語幹性）**: 左辺の末尾漢字連 $s$ について、コーパス上でその漢字連に続くひらがな連の右分岐エントロピー $H(s)$（nats）を測り、`beg_stem_max`（0.7）から `beg_noun_min`（1.8）へ線形正規化して `[0,1]` にクランプする。語幹は活用形が閉じているため低エントロピー、名詞は助詞が開いているため高エントロピーとなる。観測された漢字プレフィックスの各文字境界サフィックスにもスコアを継承させ（衝突時は最大値）、部分マージ済み左辺（`名` のみ残った `名詞` 等）でも解決できる。未観測の漢字連は名詞側 `1.0` とみなす。
2. **右ゲート `g_r` / `gate_R`（助詞性）**: 右辺の先頭ひらがな連 $h$ が付着した相異なる漢字プレフィックス数 $n(h)$ を数え、`g_r = clamp((ln n(h) - beg_particle_lo) / (beg_particle_hi - beg_particle_lo), 0, 1)` とする（既定 `beg_particle_lo = 6.0`, `beg_particle_hi = 9.0`）。`べる` のような少数の語幹に付く活用形は低く、`の` のような数万の名詞に付く助詞は `g_r \approx 1` となる。左辺が退化分布（単一フォロワー）の名詞でも右ゲートが助詞を捕捉する。

Phase 1 では `Kanji -> Hiragana` ペアのスコアを `f(l,r) * NPMI(l,r)^e`（`e = beg_strength * gate`, `NPMI` は `[0.01, 1]` にクランプ）とし、`gate` は「右側が明確な活用形（`g_r <= 0.5`）なら `g_r` のみ（右ゲート優先: `染` のように活用形が約 10 種ある動詞語幹は左エントロピーが高く `max(g_l, g_r)` で誤判定されるため）」、それ以外は `max(g_l, g_r)` とする。`beg_strength = 0` のとき ACSG の固定指数 `acsg_npmi_beta` にフォールバックし、それも 0 なら生頻度スコアとなる。さらに `gate_R >= 0.95` のペアはヒープに乗せないハード拒否 (`beg_hard_veto`) とし、大語彙でも `日本|の` が NPMI ペナルティをすり抜けないようにする。`beg_okuri_boost > 0` のとき `score * (1 + boost * (1 - g_r))` を掛け、非交差ペアでも `確か|める` 型（結合かな尾が `gate_R < 0.9`）には同ブーストを適用する。

### 3.6 外科的語彙修復 (Aligned BPE-Knockout) とタグ付きレイヤー

学習済みモデルに対し、`refine` サブコマンド（`crates/kototok-cli/src/main.rs` の `refine_model`）が gold 境界 JSONL（`text`, `byte_spans`, `okurigana_words`, `noun_particles`）を用いて 2 相の修復を行う。

**Phase A (prune)**: 予測ピースが「名詞+助詞」境界 (`noun_particles[].boundary_byte`) を内部に厳密に含む（`s < b < e`）出現を違反として、そのピースを生んだ `Merge` イベントへ帰属させる。`--prune-oku-edges` 指定時は送り仮名語の両端 (`okurigana_words[].byte_span` の `a`, `b`) も prune 境界に追加する。違反回数が `min_violations` 以上で、かつマージの両子が生存している対象に対し、`Prune { target, split_into: [left, right] }` を**その merge の直後（rank+1）に挿入**する。prune はランク位置に依存する（末尾追加では後続マージが先に target を消費してしまう）ため、常に mid-list 挿入である。挿入はランク降順で行い、既存 `tag_ranges` の添字を挿入数分シフトし、`phase2_start_rank` も調整する。全 gold 境界ではなく `noun+particle` 境界のみを対象とするのは、無害な複合語・ASCII superword の破壊を避けるためである。

**Phase B (graft)**: 予測ピースが 2 個以上で gold トークン区間をちょうど敷き詰める（under-merge）パターンを「観測分割連鎖」として収集し、`min_graft_freq` 以上のものを新規 `Merge` イベントとして `events` の**末尾**に追加する。末尾追加は常に最低優先となり、学習済み語彙と競合しない位置でのみ発火する外科的追加である。1 語あたり最大 `MAX_CHAINS = 8` の文脈依存連鎖を保持する multi-chain graft とし、prune 対象を含む連鎖は除外する。新規内部トークンは `MergedPhase2` 種別で実 `final_id` を連番付与し、部分連鎖が他文脈で表面化しても正当な ID を出力できる。

**タグ付きレイヤー（Selectable-Segmentation）**: `Model` は serde `#[serde(default)]` 互換の 2 フィールドを持ち、新旧 `.ktok` を相互ロードできる。

- `tag_ranges: Vec<TagRange{start, end, tag}>`: 出自タグ（`ja`/`casual`/`kozh`/`okuri`/`okuedge` 等）を持つ連続イベント区間。graft バッチ全体を区間として記録する。
- `tagged_prunes: Vec<(target_internal_id, tag)>`: prune は mid-list で連続区間に置けないため、`target` 内部 ID からタグへの疎な対応表で記録する。

ロード時の `filtered_by_tags(enabled)` は、無効化タグ区間内の `Merge` イベントと、無効化タグが付く `target` の `Prune` イベントを除去する。タグなし prune は常に残り、無効化マージの生成トークンは単に到達不能となる（graceful degradation）。ポリシーは決定論的で、CLI の `--mode strict`（既定・全レイヤー）、`--mode compress`（全タグ無効）、`--tags a,b`（任意選択）、および Python の `Tokenizer.load(path, tags=[...])` で切り替える。

---

## 4. 第3の柱: $O(N)$ 計算量証明と 100% 可逆性

### 4.1 最悪計算量 $O(N)$ の証明

入力文字列長を $N$（バイト数または文字数）とする。

1. **前処理 DFA**: 入力文字を先頭から末尾まで 1 パスで走査し、直前の状態と現在の文字クラスの組から定数時間 $O(1)$ で境界判定を行う。バックトラックは存在しないため、計算時間は厳密に $O(N)$、追加作業領域は $O(N)$（出力チャンク列）。
2. **Phase 1 エンコード**: `max_chunk_chars = 128` により、任意のチャンク $c$ の長さ $L_c$ は $1 \le L_c \le 128$ を満たす。チャンク内のイベント・ランク再生は高々 $O(L_c \log L_c)$（または $O(L_c^2)$）ステップで完了するが、$L_c \le 128$ が定数上界であるため 1 チャンクあたり $O(1)$ 時間である。チャンク数 $K$ は $K \le N$ であるため、Phase 1 全体の最悪時間は $\sum_{c=1}^{K} O(L_c) = O(N)$ となる。
3. **Phase 2 エンコード**: 各セグメント内のチャンク数 $M_s$ は `max_segment_chunks = 64` により $M_s \le 64$ と定数で抑えられる。よってセグメント単位の Phase 2 イベント再生も $O(1)$ であり、全セグメント合計で $O(N)$ である。
4. **帰結**: エンコード全体は最悪入力（例: `'あ'` が 1MB 連続する文字列）に対しても $O(N)$ 時間で完了し、$O(N^2)$ の劣化は起きない。

### 4.2 100% 無損失可逆性 (`decode(encode(s)) == s`)

任意の有効な UTF-8 文字列 $s$ に対して次が成り立つ。

1. DFA 分割は文字を削除・置換しないため、チャンクの連接は元の文字列と一致する（$\bigparallel_c c = s$）。
2. 初期トークン化では、ASCII はそのまま `0..128`、非 ASCII 文字は `AtomicChar`（辞書にある場合）またはその UTF-8 生バイト列に対応する `ByteFallback`（`128..256`）へ写像されるため、初期トークン列のバイト列連接は $s$ のバイト列と一致する。
3. すべての `Merge { left, right, result }` は $\text{bytes}(\text{result}) = \text{bytes}(\text{left}) \parallel \text{bytes}(\text{right})$ を満たし、すべての `Prune { target, split_into }` は $\text{bytes}(\text{target}) = \bigparallel_{u \in \text{split\_into}} \text{bytes}(u)$ を満たす。
4. よって任意のイベント再生後もバイト列の連接不変量が保たれ、`final_id_to_bytes()` により復元されるバイト列は入力 $s$ の UTF-8 バイト列と完全に一致する。したがって `decode(encode(s)) == s` が常に成立する。

---

## 5. 評価手法と 6 つのアブレーション構成

### 5.1 制御された 6 つのアブレーションプリセット

各構成要素の寄与を分離測定するため、同一学習コーパス・同一語彙サイズ（既定 `4096`）で次の 6 プリセットを訓練・比較する。

| プリセット名 | `PreTokenizerMode` | `ByteMode` | ACSG | PickyBPE (IoS) | Phase 2 | 検証対象 |
| --- | --- | --- | --- | --- | --- | --- |
| `asym-full` | `AsymScript` | `Utf8Atomic` | ON | ON | ON | 完全版 AsymScript-BPE |
| `asym-no-phase2` | `AsymScript` | `Utf8Atomic` | ON | ON | OFF | Phase 2 Superword の効果 |
| `asym-no-picky` | `AsymScript` | `Utf8Atomic` | ON | OFF | ON | PickyBPE 剪定の Dead Token 削減効果 |
| `asym-no-acsg` | `AsymScript` | `Utf8Atomic` | OFF | ON | ON | ACSG による名詞＋助詞過結合の抑制効果 |
| `strict-script` | `StrictScript` | `Utf8Atomic` | OFF | OFF | OFF | 厳格文字種分割の送り仮名分断への影響 |
| `standard-bbpe` | `StandardGpt` | `RawBytes` | OFF | OFF | OFF | 従来の標準 Byte-level BPE 対照群 |
| `global` | `AsymScript` + 多言語拡張 | `Utf8Atomic` | ON | ON | ON（16→3 階層） | 68 ドメイン多言語・数式・コード向け構成 |

加えて外部比較対象として、同一コーパス・同一語彙サイズで訓練した HuggingFace `tokenizers` の `hf-bpe` (ByteLevel BPE) と `hf-unigram` (Unigram)、ならびに大規模外部基準として `tiktoken` (`cl100k_base`, `o200k_base`) を併記する。

### 5.2 5 つの評価指標

1. **圧縮効率 (`chars/token`, `bytes/token`, `tokens`)**: 未学習テスト分割（`test_ja_prose.txt`, `test_en_prose.txt`, `test_code.txt`, `test_mixed_ja_code.txt`）ごとに計測する。
2. **無効 UTF-8 語彙率 (`invalid_utf8_rate`)**: `kototok inspect --json`（`Utf8AuditReport`）の `invalid_utf8_learned_tokens / alive_non_fallback_tokens`、および `hf-bpe` の ByteLevel デコード後の UTF-8 不正トークン率を比較する。
3. **日本語形態素境界精度 (`ja_boundary_gold.jsonl`)**:
   - **Boundary Precision / Recall / F1**: 予測トークン境界バイト集合 $B_{\text{pred}}$ と正解形態素境界バイト集合 $B_{\text{gold}}$ の一致率。
   - **Okurigana Integrity Rate (%)**: タグ付けされた活用語（例: `新しい`, `食べる`, `調べる`）のスパン $(a, b)$ 内部に分割境界が入らない割合。
   - **Noun+Particle Over-Merge Rate (%)**: タグ付けされた「名詞＋助詞」境界 $m$（例: `日本\|の`, `開発者\|が`）が予測境界 $B_{\text{pred}}$ から欠落し、1 トークンへ結合されてしまった割合（低いほど良好）。
4. **死蔵トークン率 (`Dead Token Rate %`)**: 結合済みテストコーパス全体を通して一度も出現しない語彙トークンが `vocab_size` に占める割合。
5. **処理速度と $O(N)$ 耐性**: `test_ja_prose.txt` に対するシングルスレッド処理スループット (`MB/s`) と、`adversarial_1mb.txt`（1MB 連続同一文字を含む敵対的入力）のエンコード遅延 (`ms`)。

---

## 6. 多言語 `global` 拡張・階層フォールバック Phase-2・汎用 6 形式入出力

### 6.1 多言語文字種 DFA と句読点境界 (`global` プリセット)

68 ドメイン多言語・数式・コード向けの `global` プリセットは、`PreTokenizerConfig` に次の 2 フラグを有効化する（実装: `crates/kototok-core/src/script.rs`, `pretok.rs`）。

1. **多言語文字種統合 (`multilingual_scripts = true`)**:
   - 57 言語の非 CJK 文字をスクリプト群ごとに分類し、Unicode の結合記号（`Mn` / `Mc`）およびゼロ幅接合子（`U+200C`, `U+200D`）を直前の基底文字と同一チャンクに維持してインド系・東南アジア系・アラビア系・ヘブライ系文字の分断を防ぐ。
   - 拡張ラテン文字（`U+0250..=U+02AF` の `ə` 等）およびギリシャ文字（`U+00B5` 含む）をそれぞれのスクリプト群に含める。
   - 非 CJK の全アルファベット文字種（節区切り空白を含むタイ語・ラーオ語・ミャンマー語・クメール語を含む）について、単一の先頭 ASCII 空白（`' '`）を後続語へ結合する。
2. **句読点ソフト境界拡張 (`attach_space_to_punct = true`)**:
   - ASCII 句読点の前後を Phase-2 のソフト境界として扱い、数式・コード・英語の多文字記号結合を許可する。
   - 日本語では単語と `JaPunct` の境界をハードのまま維持しつつ、隣接する `JaPunct` 同士（例: `。「`, `！！`）の間のみソフト境界とする。

### 6.2 階層フォールバック Phase-2 と省メモリ学習

`phase2_fallback_window = 3` を設定した Phase-2 集計では、最大 16 チャンクのセグメント全体が 2 回以上出現するコード行・数式をそのまま保持する一方、1 回しか出現しない長文セグメントを破棄せず 3 チャンク窓へ分割して反復句を集計する（`crates/kototok-train/src/counter.rs`）。
トレーナー実装（`crates/kototok-train/src/trainer.rs`）では Phase-1 の終了後に Phase-1 用の作業メモリを解放してから Phase-2 を集計し、シンボル列のインプレース更新と Phase-2 単発ペア事前除外を組み合わせることで、286MB・264k 語彙の学習を 4 スレッド約 36〜50 秒・ピーク RSS 1.86 GiB（スワップ 0 KiB）、780.5MB コーパスを 93.7 秒で完走する。

### 6.3 汎用 6 形式入出力 (`crates/kototok-core/src/formats/`)

`Model::load_auto` / `Model::save_auto` は、`.ktok`・kototok JSON（`json`）・Hugging Face `tokenizer.json`（`hf`）・Hugging Face ディレクトリ（`hf-dir`）・OpenAI `.tiktoken`（`tiktoken`）・GPT-2 形式（`bpe`）の 6 形式を統一的に扱う。
Hugging Face 形式の書き出しでは、UTF-8 原子文字の中間バイト列と PickyBPE 剪定済み祖先を ByteLevel BPE 語彙として補完し、密な ID 空間（`0..V-1`）および `AsymScript` / `StrictScript` / `StandardGpt` + `global` フラグを反映した `pre_tokenizer` 正規表現を出力する。これにより、Hugging Face `tokenizers` ランタイムと kototok 再読込の双方で完全可逆エンコードを再現できる。
