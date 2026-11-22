#!/usr/bin/env python3
"""Fetch Aozora Bunko texts as extra Japanese training corpus.

Reads the official catalog (list_person_all_extended_utf8.csv), selects
public-domain 新字新仮名 works spread across authors, downloads the text
zips, decodes Shift-JIS, strips Aozora markup, and writes docs joined by
\n\n to train_corpus_extra.txt.

Usage: ./.venv/bin/python aozora_fetch.py --mb 30 --per-author 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "artifacts_real" / "data"
CACHE_DIR = EVAL_DIR / "artifacts_real" / ".fetch_cache"
UA = "kototok-eval/0.1 (local tokenizer benchmark; contact: local)"

RE_SEP = re.compile(r"^-{5,}\s*$", re.M)
RE_RUBY = re.compile(r"《[^》]*》")
RE_NOTE = re.compile(r"［＃[^］]*］")
RE_PIPE = re.compile(r"[｜]")
RE_WS = re.compile(r"[ \u3000]+")
MIN_DOC_BYTES = 5_000


def cache_path(url: str) -> Path:
    return CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".bin")


def fetch(url: str, retries: int = 3) -> bytes | None:
    cp = cache_path(url)
    if cp.exists():
        return cp.read_bytes()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            cp.write_bytes(data)
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    print(f"[warn] {url}: {last}", file=sys.stderr)
    return None


def clean_aozora(text: str) -> str:
    # Aozora layout: header --sep-- [symbol legend] --sep-- BODY [--sep--]
    # colophon. Some files omit the trailing separator, so the body is the
    # largest non-legend segment after the first separator, with the 底本
    # colophon cut off near the end.
    parts = RE_SEP.split(text)
    cands = [p for p in parts[1:]
             if p.strip() and "記号について" not in p[:200]]
    body = max(cands, key=len) if cands else parts[-1]
    m = re.search(r"\n底本：", body)
    if m and m.start() > len(body) // 2:
        body = body[:m.start()]
    body = RE_RUBY.sub("", body)
    body = RE_NOTE.sub("", body)
    body = RE_PIPE.sub("", body)
    lines = [RE_WS.sub("", ln).strip() for ln in body.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/list_person_all_extended_utf8.csv")
    ap.add_argument("--mb", type=float, default=30.0)
    ap.add_argument("--per-author", type=int, default=8)
    ap.add_argument("--out", default="train_corpus_extra.txt")
    args = ap.parse_args()

    rows = list(csv.reader(open(args.csv, encoding="utf-8-sig")))
    hdr = rows[0]
    i_url = hdr.index("テキストファイルURL")
    i_kana = hdr.index("文字遣い種別")
    i_right = hdr.index("作品著作権フラグ")
    i_author = hdr.index("姓") + 0  # use 人物ID for grouping
    i_pid = hdr.index("人物ID")
    i_title = hdr.index("作品名")

    by_author: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in rows[1:]:
        if len(r) <= i_url or not r[i_url].endswith(".zip"):
            continue
        if r[i_kana] != "新字新仮名" or r[i_right] != "なし":
            continue
        by_author[r[i_pid]].append((r[i_title], r[i_url]))

    print(f"[debug] rows={len(rows)} authors={len(by_author)}",
          file=sys.stderr, flush=True)
    # round-robin across authors for diversity
    picks: list[tuple[str, str]] = []
    for k in range(args.per_author):
        for works in by_author.values():
            if k < len(works):
                picks.append(works[k])

    target = int(args.mb * 1024 * 1024)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_doc(item: tuple[str, str]) -> tuple[str, str] | None:
        title, url = item
        raw = fetch(url)
        if not raw:
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".txt"))
                text = zf.read(name).decode("cp932", errors="replace")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] zip {url.rsplit('/', 1)[-1]}: {e}", file=sys.stderr)
            return None
        body = clean_aozora(text)
        if len(body.encode("utf-8")) < MIN_DOC_BYTES:
            return None
        return title, body

    docs: list[str] = []
    total = 0
    batch = 36
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i in range(0, len(picks), batch):
            if total >= target:
                break
            futs = [pool.submit(get_doc, it) for it in picks[i:i + batch]]
            for f in futs:
                res = f.result()
                if res is None:
                    continue
                title, body = res
                docs.append(body)
                total += len(body.encode("utf-8"))
                print(f"[aozora] {total / 1e6:.1f}MB/{args.mb}"
                      f"  ({len(docs)} docs, {title[:16]})",
                      file=sys.stderr, flush=True)
                if total >= target:
                    break

    out = DATA_DIR / args.out
    out.write_text("\n\n".join(docs) + "\n", encoding="utf-8")
    print(f"[done] {out}  {total / 1e6:.1f}MB  {len(docs)} docs", file=sys.stderr)


if __name__ == "__main__":
    main()
