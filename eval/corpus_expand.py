#!/usr/bin/env python3
"""Expand the real training corpus with bulk Japanese Wikipedia extracts.

Walks the jawiki article index alphabetically via generator=allpages, pulling
plain-text extracts 500 pages per API request. Stubs (<2KB) are dropped.
Output: train_corpus_extra.txt (docs joined by \\n\\n), leaving
train_corpus.txt untouched.

Usage: ./.venv/bin/python corpus_expand.py --mb 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "artifacts_real" / "data"
CACHE_DIR = EVAL_DIR / "artifacts_real" / ".fetch_cache"
UA = "kototok-eval/0.1 (local tokenizer benchmark; contact: local)"

API = "https://ja.wikipedia.org/w/api.php"
MIN_DOC_BYTES = 2_000


def cache_path(url: str) -> Path:
    return CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")


def fetch_json(url: str, retries: int = 4) -> dict | None:
    cp = cache_path(url)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            cp.unlink(missing_ok=True)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            cp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    print(f"[warn] fetch failed: {last}", file=sys.stderr)
    return None


def pages_batch(cont: dict | None, limit: int = 20) -> tuple[list[str], dict | None]:
    """Allpages+extracts walk. `cont` carries the full continuation params."""
    params: dict[str, str] = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": str(limit),
        "generator": "allpages",
        "gapnamespace": "0",
        "gaplimit": str(limit),
    }
    if cont:
        params.update(cont)
        params.setdefault("continue", "||")
    url = API + "?" + urllib.parse.urlencode(params)
    data = fetch_json(url)
    if not data:
        return [], cont
    docs = []
    for page in data.get("query", {}).get("pages", []):
        text = page.get("extract", "").strip()
        if len(text.encode("utf-8")) >= MIN_DOC_BYTES:
            docs.append(text)
    nxt = data.get("continue")
    if nxt:
        nxt = {k: str(v) for k, v in nxt.items() if k != "continue"}
    return docs, nxt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=40.0, help="target EXTRA MiB")
    ap.add_argument("--out", default="train_corpus_extra.txt")
    ap.add_argument("--from", dest="gapfrom", default=None,
                    help="start article title (default: beginning of index)")
    args = ap.parse_args()

    target = int(args.mb * 1024 * 1024)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    docs: list[str] = []
    total = 0
    cont: dict | None = {"gapfrom": args.gapfrom} if args.gapfrom else {}
    empty_streak = 0

    while total < target and empty_streak < 40 and cont is not None:
        batch, cont = pages_batch(cont if cont else None)
        if not batch:
            empty_streak += 1
            time.sleep(0.5)
        else:
            empty_streak = 0
            for d in batch:
                docs.append(d)
                total += len(d.encode("utf-8"))
        print(f"[expand] {total / 1e6:.1f}MB / {args.mb}MiB  ({len(docs)} docs)",
              file=sys.stderr, flush=True)
        time.sleep(0.3)

    out = DATA_DIR / args.out
    out.write_text("\n\n".join(docs) + "\n", encoding="utf-8")
    print(f"[done] {out}  {total / 1e6:.1f}MB  {len(docs)} docs", file=sys.stderr)


if __name__ == "__main__":
    main()
