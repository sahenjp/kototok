#!/usr/bin/env python3
"""Download license-clean REAL text corpus for kototok eval.

Sources (fixed URL lists, deterministic order):
  - JA:  Japanese Wikipedia raw wikitext (CC BY-SA), markup-stripped.
  - EN:  Project Gutenberg UTF-8 plain-text books (public domain).
  - Code: rust-lang/rust library sources (MIT/Apache) + python/cpython Lib (PSF).
  - Mixed: real OSS docs/READMEs + JA wiki paragraphs embedded with code fences.
  - adversarial_1mb.txt / ja_boundary_gold.jsonl: copied verbatim from
    artifacts/data (synthetic corpus) so gold/adversarial stay comparable.

Outputs to artifacts_real/data/ and prints a manifest JSON to stdout.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "artifacts_real" / "data"
SYNTH_DATA_DIR = EVAL_DIR / "artifacts" / "data"

UA = "kototok-eval/0.1 (local tokenizer benchmark; contact: local)"
SEED = 20261122
CACHE_DIR = EVAL_DIR / "artifacts_real" / ".fetch_cache"

# Byte targets — fail loudly if unmet.
JA_TRAIN_MIN = 7_000_000
JA_TRAIN_STOP = 8_500_000  # stop early once comfortably above target
EN_TRAIN_MIN = 2_500_000
CODE_TRAIN_MIN = 1_500_000
MIXED_TOTAL_MIN = 1_000_000  # train + test

# ---------------------------------------------------------------------------
# Japanese Wikipedia titles (action=raw). Fetched in this order.
# ---------------------------------------------------------------------------

# 5 held-out titles -> test_ja_prose.txt (fetched first so they're guaranteed)
JA_TEST_TITLES = [
    "俳句",
    "将棋",
    "気候変動",
    "量子コンピュータ",
    "新幹線",
]

# 2 titles embedded with code fences -> mixed train docs
JA_MIXED_TRAIN_TITLES = [
    "オープンソースソフトウェア",
    "ソフトウェア開発",
]

# 1 title -> mixed test doc (first that fetches)
JA_MIXED_TEST_TITLES = [
    "バージョン管理システム",
    "ソフトウェアテスト",
]

# ~90 large, diverse train articles
JA_TRAIN_TITLES = [
    "日本", "東京", "大阪", "京都", "名古屋", "北海道", "沖縄県", "福岡",
    "日本の歴史", "江戸時代", "明治維新", "第二次世界大戦", "日本国憲法", "戦国時代 (日本)",
    "日本語", "漢字", "ひらがな", "カタカナ", "日本文学", "夏目漱石",
    "歌舞伎", "茶道", "武士", "神道", "仏教", "キリスト教", "宗教", "哲学",
    "心理学", "言語学", "物理学", "量子力学", "相対性理論", "化学", "生物学",
    "進化", "遺伝学", "細胞", "ウイルス", "免疫学", "医学",
    "数学", "統計学", "経済学", "経済", "貿易", "法律", "民主主義", "教育",
    "大学", "音楽", "クラシック音楽", "ジャズ", "料理", "和食", "鉄道",
    "自動車", "航空機", "建築", "都市計画", "コンピュータ", "インターネット",
    "人工知能", "機械学習", "自然言語処理", "プログラミング", "プログラミング言語",
    "アルゴリズム", "データベース", "オペレーティングシステム", "半導体", "ロボット",
    "電気工学", "暗号理論", "太陽系", "宇宙", "地球", "海洋学", "気象学",
    "地震", "火山", "恐竜", "農業", "工業", "エネルギー", "環境問題",
    "造船", "材料工学", "囲碁", "ポケットモンスター", "アニメ", "漫画",
    "世界遺産", "オリンピック", "江戸", "奈良", "平安時代",
    "鎌倉時代", "室町時代", "弥生時代", "縄文時代", "昭和時代", "南北朝時代 (日本)",
    "イギリス", "アメリカ合衆国", "中華人民共和国", "ドイツ", "フランス", "ロシア",
    "大韓民国", "インド", "ブラジル", "オーストラリア", "カナダ", "イタリア",
    "第一次世界大戦", "冷戦", "産業革命", "フランス革命", "ローマ帝国", "古代エジプト",
    "ヨーロッパ", "アジア", "アフリカ", "太平洋", "富士山", "ヒマラヤ",
    "サッカー", "野球", "大相撲", "柔道", "映画", "テレビ", "新聞", "写真",
    "絵画", "彫刻", "演劇", "オペラ", "ピアノ", "ヴァイオリン", "電気",
    "原子", "分子", "元素", "周期表", "タンパク質", "脳", "心臓", "植物",
    "動物", "哺乳類", "鳥類", "昆虫", "細菌", "生態系", "森林", "砂漠",
    "台風", "津波", "洪水", "栄養学", "公衆衛生", "医療", "薬学",
    "土木工学", "機械工学", "化学工学", "情報工学", "衛星", "携帯電話",
    "テレビゲーム", "将棋盤", "鉄道駅", "空港", "高速道路", "橋",
]

# ---------------------------------------------------------------------------
# Project Gutenberg (public-domain English prose). Last two -> test.
# ---------------------------------------------------------------------------
GUTENBERG_IDS = [1342, 84, 98, 1661, 2600, 2701, 174, 345, 11, 1952]
GUTENBERG_HOSTS = [
    "https://www.gutenberg.org",
    "https://gutenberg.pglaf.org",
]

# ---------------------------------------------------------------------------
# Code files (rust-lang/rust MIT/Apache; python/cpython PSF).
# Order matters: the last 8 *successfully fetched* go to test_code.txt.
# ---------------------------------------------------------------------------
_GH = "https://raw.githubusercontent.com"
RUST_BASE = f"{_GH}/rust-lang/rust/master"
PY_BASE = f"{_GH}/python/cpython/main"

CODE_URLS = [
    # Rust std/core/alloc
    f"{RUST_BASE}/library/std/src/lib.rs",
    f"{RUST_BASE}/library/std/src/error.rs",
    f"{RUST_BASE}/library/std/src/fs.rs",
    f"{RUST_BASE}/library/std/src/path.rs",
    f"{RUST_BASE}/library/std/src/process.rs",
    f"{RUST_BASE}/library/std/src/time.rs",
    f"{RUST_BASE}/library/std/src/panicking.rs",
    f"{RUST_BASE}/library/std/src/f32.rs",
    f"{RUST_BASE}/library/std/src/f64.rs",
    f"{RUST_BASE}/library/std/src/io/mod.rs",
    f"{RUST_BASE}/library/std/src/thread/mod.rs",
    f"{RUST_BASE}/library/core/src/lib.rs",
    f"{RUST_BASE}/library/core/src/num/mod.rs",
    f"{RUST_BASE}/library/core/src/str/mod.rs",
    f"{RUST_BASE}/library/core/src/slice/mod.rs",
    f"{RUST_BASE}/library/core/src/iter/mod.rs",
    f"{RUST_BASE}/library/core/src/fmt/mod.rs",
    f"{RUST_BASE}/library/core/src/ops/mod.rs",
    f"{RUST_BASE}/library/core/src/option.rs",
    f"{RUST_BASE}/library/core/src/result.rs",
    f"{RUST_BASE}/library/alloc/src/string.rs",
    f"{RUST_BASE}/library/alloc/src/vec/mod.rs",
    f"{RUST_BASE}/library/alloc/src/boxed.rs",
    f"{RUST_BASE}/library/alloc/src/rc.rs",
    f"{RUST_BASE}/library/alloc/src/collections/btree/map.rs",
    # Python stdlib
    f"{PY_BASE}/Lib/argparse.py",
    f"{PY_BASE}/Lib/ast.py",
    f"{PY_BASE}/Lib/dataclasses.py",
    f"{PY_BASE}/Lib/difflib.py",
    f"{PY_BASE}/Lib/enum.py",
    f"{PY_BASE}/Lib/functools.py",
    f"{PY_BASE}/Lib/heapq.py",
    f"{PY_BASE}/Lib/inspect.py",
    f"{PY_BASE}/Lib/io.py",
    f"{PY_BASE}/Lib/ipaddress.py",
    f"{PY_BASE}/Lib/os.py",
    f"{PY_BASE}/Lib/pathlib/__init__.py",
    f"{PY_BASE}/Lib/pickle.py",
    f"{PY_BASE}/Lib/queue.py",
    f"{PY_BASE}/Lib/random.py",
    f"{PY_BASE}/Lib/shlex.py",
    f"{PY_BASE}/Lib/shutil.py",
    f"{PY_BASE}/Lib/socket.py",
    f"{PY_BASE}/Lib/ssl.py",
    f"{PY_BASE}/Lib/statistics.py",
    f"{PY_BASE}/Lib/string.py",
    f"{PY_BASE}/Lib/tempfile.py",
    f"{PY_BASE}/Lib/textwrap.py",
    f"{PY_BASE}/Lib/threading.py",
    f"{PY_BASE}/Lib/traceback.py",
    f"{PY_BASE}/Lib/types.py",
    f"{PY_BASE}/Lib/typing.py",
    f"{PY_BASE}/Lib/zipfile.py",
    f"{PY_BASE}/Lib/collections/__init__.py",
    f"{PY_BASE}/Lib/json/decoder.py",
    f"{PY_BASE}/Lib/json/encoder.py",
    f"{PY_BASE}/Lib/logging/__init__.py",
    f"{PY_BASE}/Lib/re/__init__.py",
    f"{PY_BASE}/Lib/re/_compiler.py",
    f"{PY_BASE}/Lib/re/_parser.py",
    f"{PY_BASE}/Lib/unittest/case.py",
    f"{PY_BASE}/Lib/urllib/parse.py",
    f"{PY_BASE}/Lib/urllib/request.py",
    f"{PY_BASE}/Lib/xml/etree/ElementTree.py",
    f"{PY_BASE}/Lib/email/message.py",
    f"{PY_BASE}/Lib/email/parser.py",
    # ---- last 8 fetched -> test_code.txt (mix of rust+python at the tail) ----
    f"{PY_BASE}/Lib/configparser.py",
    f"{RUST_BASE}/library/std/src/rt.rs",
    f"{PY_BASE}/Lib/csv.py",
    f"{RUST_BASE}/library/alloc/src/sync.rs",
    f"{PY_BASE}/Lib/contextlib.py",
    f"{RUST_BASE}/library/core/src/future.rs",
    f"{PY_BASE}/Lib/copy.py",
    f"{PY_BASE}/Lib/tokenize.py",
    f"{PY_BASE}/Lib/uuid.py",
    f"{PY_BASE}/Lib/warnings.py",
]

# ---------------------------------------------------------------------------
# Mixed-language real docs (markdown/rst). Last two fetched -> test_mixed.
# ---------------------------------------------------------------------------
MIXED_URLS = [
    f"{RUST_BASE}/RELEASES.md",
    f"{_GH}/BurntSushi/ripgrep/master/GUIDE.md",
    f"{_GH}/BurntSushi/ripgrep/master/FAQ.md",
    f"{_GH}/BurntSushi/ripgrep/master/CHANGELOG.md",
    f"{_GH}/rust-lang/book/main/src/ch02-00-guessing-game-tutorial.md",
    f"{_GH}/rust-lang/book/main/src/ch04-01-what-is-ownership.md",
    f"{_GH}/rust-lang/book/main/src/ch08-01-vectors.md",
    f"{_GH}/rust-lang/book/main/src/ch09-02-recoverable-errors-with-result.md",
    f"{_GH}/rust-lang/book/main/src/ch13-01-closures.md",
    f"{_GH}/rust-lang/book/main/src/ch00-00-introduction.md",
    f"{RUST_BASE}/README.md",
    f"{_GH}/tokio-rs/tokio/master/README.md",
    f"{_GH}/tokio-rs/tokio/master/CONTRIBUTING.md",
    f"{_GH}/serde-rs/serde/master/README.md",
    f"{PY_BASE}/README.rst",
    f"{PY_BASE}/Doc/tutorial/introduction.rst",
    f"{PY_BASE}/Doc/tutorial/controlflow.rst",
    f"{PY_BASE}/Doc/tutorial/datastructures.rst",
    f"{PY_BASE}/Doc/tutorial/classes.rst",
    # ---- tail: held out for test ----
    f"{_GH}/BurntSushi/ripgrep/master/README.md",
    f"{_GH}/rust-lang/book/main/src/ch01-02-hello-world.md",
]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
def _cache_path(url: str) -> Path:
    return CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".bin")


def fetch(url: str, retries: int = 3, sleep: float = 0.3) -> bytes | None:
    cp = _cache_path(url)
    if cp.exists():
        return cp.read_bytes()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if sleep:
                time.sleep(sleep)
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cp.write_bytes(data)
            except OSError:
                pass
            return data
        except Exception as e:  # noqa: BLE001 - log and retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"[warn] fetch failed {url}: {last_err}", file=sys.stderr)
    return None


def fetch_text(url: str, **kw) -> str | None:
    raw = fetch(url, **kw)
    if raw is None:
        return None
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Wikitext cleaner
# ---------------------------------------------------------------------------
_RE_REF_SELF = re.compile(r"<ref\b[^>]*/>", re.I)
_RE_REF_BLOCK = re.compile(r"<ref\b[^>]*>.*?</ref>", re.I | re.S)
_RE_TAGBLOCK = re.compile(
    r"<(math|gallery|nowiki|timeline|syntaxhighlight|source|pre|score|chem|"
    r"mapframe|maplink|templatedata|graph|hiero|imagemap|inputbox|poem|ref"
    r"erences|section|table)\b[^>]*>.*?</\1>",
    re.I | re.S,
)
_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_TEMPLATE_INNER = re.compile(r"\{\{[^{}]*\}\}")
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_RE_NEST = r"[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*"
_RE_FILELINK = re.compile(
    r"\[\[(?:Category|カテゴリ|File|ファイル|画像|Image|Media|メディア):"
    + _RE_NEST
    + r"\]\]",
    re.I,
)
_NS = (
    r"(?:Wikipedia|ウィキペディア|WP|Help|ヘルプ|H|Portal|ポータル|特別|利用者|"
    r"User|ノート|Talk|トーク|Template|テンプレート|MediaWiki|メディアウィキ|"
    r"モジュール|Module|CAT|[a-z]{2,3}(?:-[a-zA-Z]+)?)"
)
_RE_NSLINK = re.compile(r"\[\[" + _NS + r":" + _RE_NEST + r"\]\]")
_RE_PIPED_LINK = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
_RE_PLAIN_LINK = re.compile(r"\[\[([^\]|]*)\]\]")
_RE_EXT_LABEL = re.compile(r"\[(?:https?:)?//[^\s\]]+\s+([^\]]*)\]")
_RE_EXT_BARE = re.compile(r"\[(?:https?:)?//[^\s\]]+\]")
_RE_HEADER = re.compile(r"^(={1,6})\s*(.*?)\s*\1\s*$", re.M)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_LISTMARK = re.compile(r"^[*#;: ]+")
_RE_NUM_ENTITY = re.compile(r"&#(x[0-9a-fA-F]+|\d+);")


def _num_entity(m: re.Match) -> str:
    g = m.group(1)
    try:
        cp = int(g[1:], 16) if g.startswith("x") else int(g)
        return chr(cp)
    except (ValueError, OverflowError):
        return ""


def clean_wikitext(text: str) -> str:
    text = _RE_COMMENT.sub("", text)
    text = _RE_REF_SELF.sub("", text)
    text = _RE_REF_BLOCK.sub("", text)
    text = _RE_TAGBLOCK.sub("", text)
    # nested templates: iterate to fixpoint
    for _ in range(12):
        new = _RE_TEMPLATE_INNER.sub("", text)
        if new == text:
            break
        text = new
    text = text.replace("{{", "").replace("}}", "")
    text = _RE_TABLE.sub("", text)
    text = _RE_FILELINK.sub("", text)
    text = _RE_NSLINK.sub("", text)
    for _ in range(6):
        new = _RE_PIPED_LINK.sub(r"\1", text)
        if new == text:
            break
        text = new
    text = _RE_PLAIN_LINK.sub(r"\1", text)
    text = _RE_EXT_LABEL.sub(r"\1", text)
    text = _RE_EXT_BARE.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = _RE_HEADER.sub(r"\2", text)
    text = _RE_TAG.sub("", text)
    text = html.unescape(_RE_NUM_ENTITY.sub(_num_entity, text))

    lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("__", "{|", "|-", "|}", "|", "!", "{", "}")):
            continue
        line = _RE_LISTMARK.sub("", line).strip()
        if len(line) >= 2:
            lines.append(line)
    return "\n".join(lines)


def wiki_raw_url(title: str) -> str:
    return (
        "https://ja.wikipedia.org/w/index.php?title="
        + urllib.parse.quote(title)
        + "&action=raw"
    )


# ---------------------------------------------------------------------------
# Gutenberg cleaner
# ---------------------------------------------------------------------------
_RE_GUT_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.I)
_RE_GUT_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)


def strip_gutenberg(text: str) -> str:
    m = _RE_GUT_START.search(text)
    if m:
        text = text[m.end() :]
    m = _RE_GUT_END.search(text)
    if m:
        text = text[: m.start()]
    return text.strip() + "\n"


def gutenberg_urls(book_id: int) -> list[str]:
    urls = [f"{h}/cache/epub/{book_id}/pg{book_id}.txt" for h in GUTENBERG_HOSTS]
    digits = str(book_id)
    path = "/".join(digits[:-1])
    for h in GUTENBERG_HOSTS:
        urls.append(f"{h}/{path}/{book_id}/{book_id}-0.txt")
        urls.append(f"{h}/{path}/{book_id}/{book_id}.txt")
    return urls


def fetch_gutenberg(book_id: int) -> str | None:
    for url in gutenberg_urls(book_id):
        raw = fetch(url, retries=2, sleep=0.3)
        if raw is None or len(raw) < 10_000:
            continue
        text = raw.decode("utf-8-sig", errors="replace")
        stripped = strip_gutenberg(text)
        if len(stripped.encode("utf-8")) > 10_000:
            return stripped
    return None


# ---------------------------------------------------------------------------
# JA + code-fence embedded doc builder
# ---------------------------------------------------------------------------
def build_ja_code_doc(title: str, wiki_text: str, code_chunks: list[tuple[str, str]], n_paras: int = 60) -> str:
    paras = [p for p in wiki_text.split("\n") if p.strip()][:n_paras]
    out = [f"# {title} についての技術ノート", ""]
    ci = 0
    for i, para in enumerate(paras):
        out.append(para)
        out.append("")
        if i % 4 == 3 and code_chunks:
            lang, chunk = code_chunks[ci % len(code_chunks)]
            ci += 1
            out.append(f"```{lang}")
            out.append(chunk.rstrip())
            out.append("```")
            out.append("")
    return "\n".join(out)


def code_chunk_of(text: str, n_lines: int = 50) -> str:
    lines = text.split("\n")
    return "\n".join(lines[:n_lines])


def lang_of(url: str) -> str:
    return "rust" if url.endswith(".rs") else "python"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def u8len(s: str) -> int:
    return len(s.encode("utf-8"))


def fail(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # ---------------- JA: Wikipedia ----------------
    ja_test_texts: dict[str, str] = {}
    ja_mixed_train_texts: dict[str, str] = {}
    ja_mixed_test_texts: dict[str, str] = {}
    ja_train_texts: list[str] = []
    ja_train_bytes = 0

    def fetch_wiki(title: str) -> str | None:
        raw = fetch_text(wiki_raw_url(title), retries=3, sleep=0.5)
        if raw is None:
            return None
        cleaned = clean_wikitext(raw)
        return cleaned if u8len(cleaned) > 2_000 else None

    # held-out JA first (guaranteed fetched)
    for t in JA_TEST_TITLES:
        c = fetch_wiki(t)
        if c:
            ja_test_texts[t] = c
        else:
            failures.append(f"ja:test:{t}")
    for t in JA_MIXED_TRAIN_TITLES:
        c = fetch_wiki(t)
        if c:
            ja_mixed_train_texts[t] = c
        else:
            failures.append(f"ja:mixed_train:{t}")
    for t in JA_MIXED_TEST_TITLES:
        c = fetch_wiki(t)
        if c:
            ja_mixed_test_texts[t] = c
            break
        failures.append(f"ja:mixed_test:{t}")

    for t in JA_TRAIN_TITLES:
        if ja_train_bytes >= JA_TRAIN_STOP:
            break
        c = fetch_wiki(t)
        if c:
            ja_train_texts.append(c)
            ja_train_bytes += u8len(c)
        else:
            failures.append(f"ja:train:{t}")

    if not ja_test_texts:
        fail("no JA test articles fetched (Wikipedia unreachable?)")
    if ja_train_bytes < JA_TRAIN_MIN:
        fail(f"JA train {ja_train_bytes}B < {JA_TRAIN_MIN}B after {len(ja_train_texts)} articles")
    print(f"[ja] train {ja_train_bytes}B/{len(ja_train_texts)} articles, "
          f"test {sum(u8len(v) for v in ja_test_texts.values())}B/{len(ja_test_texts)}", file=sys.stderr)

    # ---------------- EN: Gutenberg ----------------
    en_train_texts: list[str] = []
    en_test_texts: list[str] = []
    for book_id in GUTENBERG_IDS:
        text = fetch_gutenberg(book_id)
        if text is None:
            failures.append(f"gutenberg:{book_id}")
            continue
        if len(en_test_texts) < 2 and book_id in GUTENBERG_IDS[-2:]:
            en_test_texts.append(text)
        else:
            en_train_texts.append(text)
    en_train_bytes = sum(u8len(t) for t in en_train_texts)
    if not en_test_texts:
        fail("no EN test books fetched (Gutenberg unreachable?)")
    if en_train_bytes < EN_TRAIN_MIN:
        fail(f"EN train {en_train_bytes}B < {EN_TRAIN_MIN}B")
    print(f"[en] train {en_train_bytes}B/{len(en_train_texts)} books, "
          f"test {sum(u8len(t) for t in en_test_texts)}B/{len(en_test_texts)}", file=sys.stderr)

    # ---------------- Code: GitHub raw ----------------
    code_docs: list[tuple[str, str]] = []  # (url, text)
    for url in CODE_URLS:
        text = fetch_text(url, retries=2, sleep=0.2)
        if text is None or u8len(text) < 1_000:
            failures.append(f"code:{url.rsplit('/', 1)[-1]}")
            continue
        code_docs.append((url, text))
    if len(code_docs) <= 9:
        fail(f"only {len(code_docs)} code files fetched (GitHub unreachable?)")
    code_test = code_docs[-8:]
    code_train = code_docs[:-8]
    code_train_bytes = sum(u8len(t) for _, t in code_train)
    if code_train_bytes < CODE_TRAIN_MIN:
        fail(f"code train {code_train_bytes}B < {CODE_TRAIN_MIN}B")
    print(f"[code] train {code_train_bytes}B/{len(code_train)} files, "
          f"test {sum(u8len(t) for _, t in code_test)}B/{len(code_test)}", file=sys.stderr)

    # ---------------- Mixed: docs ----------------
    mixed_docs: list[tuple[str, str]] = []
    for url in MIXED_URLS:
        text = fetch_text(url, retries=2, sleep=0.2)
        if text is None or u8len(text) < 500:
            failures.append(f"mixed:{url.rsplit('/', 1)[-1]}")
            continue
        mixed_docs.append((url, text))
    if len(mixed_docs) < 3:
        fail(f"only {len(mixed_docs)} mixed docs fetched")
    mixed_test = mixed_docs[-2:]
    mixed_train = [(u, t) for u, t in mixed_docs[:-2]]

    # JA-embedded docs: JA wiki paragraphs + fenced code chunks
    train_chunks = [(lang_of(u), code_chunk_of(t)) for u, t in code_train]
    test_chunks = [(lang_of(u), code_chunk_of(t)) for u, t in code_test]
    for title, wiki in ja_mixed_train_texts.items():
        mixed_train.append((f"jawiki+code:{title}", build_ja_code_doc(title, wiki, train_chunks)))
    for title, wiki in ja_mixed_test_texts.items():
        mixed_test.append((f"jawiki+code:{title}", build_ja_code_doc(title, wiki, test_chunks)))

    mixed_total = sum(u8len(t) for _, t in mixed_train) + sum(u8len(t) for _, t in mixed_test)
    if mixed_total < MIXED_TOTAL_MIN:
        fail(f"mixed total {mixed_total}B < {MIXED_TOTAL_MIN}B")
    print(f"[mixed] train {sum(u8len(t) for _, t in mixed_train)}B/{len(mixed_train)} docs, "
          f"test {sum(u8len(t) for _, t in mixed_test)}B/{len(mixed_test)}", file=sys.stderr)

    # ---------------- Write test splits ----------------
    test_ja = "\n\n".join(ja_test_texts.values()) + "\n"
    test_en = "\n\n".join(en_test_texts) + "\n"
    test_code = "\n\n".join(t for _, t in code_test) + "\n"
    test_mixed = "\n\n".join(t for _, t in mixed_test) + "\n"
    for name, content in [
        ("test_ja_prose.txt", test_ja),
        ("test_en_prose.txt", test_en),
        ("test_code.txt", test_code),
        ("test_mixed_ja_code.txt", test_mixed),
    ]:
        if u8len(content) < 10_000:
            fail(f"{name} too small ({u8len(content)}B)")
        (DATA_DIR / name).write_text(content, encoding="utf-8")

    # ---------------- Interleave train corpus ----------------
    rng = random.Random(SEED)
    train_blocks: list[str] = []
    for t in ja_train_texts:
        train_blocks.append(t.rstrip("\n"))
    for t in en_train_texts:
        train_blocks.append(t.rstrip("\n"))
    for _, t in code_train:
        train_blocks.append(t.rstrip("\n"))
    for _, t in mixed_train:
        train_blocks.append(t.rstrip("\n"))
    rng.shuffle(train_blocks)
    train_text = "\n\n".join(train_blocks) + "\n"
    (DATA_DIR / "train_corpus.txt").write_text(train_text, encoding="utf-8")

    # ---------------- Verbatim copies ----------------
    for name in ("adversarial_1mb.txt", "ja_boundary_gold.jsonl"):
        src = SYNTH_DATA_DIR / name
        if not src.exists():
            fail(f"missing {src} (run corpus_builder first)")
        shutil.copyfile(src, DATA_DIR / name)

    # ---------------- Manifest ----------------
    manifest = {
        "data_dir": str(DATA_DIR),
        "files": {},
        "sources": {
            "ja_wiki_train_articles": len(ja_train_texts),
            "ja_wiki_test_articles": len(ja_test_texts),
            "ja_wiki_mixed_articles": len(ja_mixed_train_texts) + len(ja_mixed_test_texts),
            "gutenberg_train_books": len(en_train_texts),
            "gutenberg_test_books": len(en_test_texts),
            "code_train_files": len(code_train),
            "code_test_files": len(code_test),
            "mixed_train_docs": len(mixed_train),
            "mixed_test_docs": len(mixed_test),
        },
        "splits_bytes": {
            "ja_train": ja_train_bytes,
            "en_train": en_train_bytes,
            "code_train": code_train_bytes,
            "mixed_train": sum(u8len(t) for _, t in mixed_train),
            "test_ja": u8len(test_ja),
            "test_en": u8len(test_en),
            "test_code": u8len(test_code),
            "test_mixed": u8len(test_mixed),
        },
        "fetch_failures": failures,
    }
    for fname in [
        "train_corpus.txt",
        "test_ja_prose.txt",
        "test_en_prose.txt",
        "test_code.txt",
        "test_mixed_ja_code.txt",
        "adversarial_1mb.txt",
        "ja_boundary_gold.jsonl",
    ]:
        p = DATA_DIR / fname
        raw = p.read_bytes()
        manifest["files"][fname] = {
            "bytes": len(raw),
            "sha256_prefix": hashlib.sha256(raw).hexdigest()[:12],
        }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
