#!/usr/bin/env python3
"""Fetch English Wikipedia text for corpus expansion (HF wikimedia/wikipedia).

Downloads one parquet shard of the 20231101.en dump from Hugging Face and
writes plain article text (docs joined by \\n\\n) to stdout or --output.
Requires pyarrow (uv pip install pyarrow).

Usage: ./.venv/bin/python enwiki_fetch.py --mb 200 --output enwiki.txt
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SHARD_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "20231101.en/train-00009-of-00041.parquet"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=200)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shard", default=SHARD_URL)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    local = Path("/tmp/enwiki_shard.parquet")
    if not local.exists():
        print(f"[fetch] {args.shard}", flush=True)
        urllib.request.urlretrieve(args.shard, local)
    texts = pq.read_table(local, columns=["text"]).column("text").to_pylist()

    out, tot = [], 0
    for tx in texts:
        out.append(tx)
        tot += len(tx.encode("utf-8"))
        if tot > args.mb * 1024 * 1024:
            break
    args.output.write_text("\n\n".join(out), encoding="utf-8")
    print(f"[done] {args.output} docs={len(out)} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
