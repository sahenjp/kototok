"""Compression and encode-speed bench of kototok against production tokenizers, including the
Inkling 2 Mini tokenizer of the model this machine serves.

    cd eval && .venv/bin/python tokenizer_bench.py [--models a,b] [--sets ja_prose,en_prose]

Reads the same held-out test sets `run_eval.py` uses (`artifacts_real/data/test_*.txt`) and the
external tokenizers `artifacts_real/external/`, reports tokens, chars/token, bytes/token and an
encode rate in MB/s on each set. kototok models load through the pylib binding, so this needs the
Python the binding was built for (cp313).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "artifacts_real" / "data"
EXTERNAL_DIR = EVAL_DIR / "artifacts_real" / "external"
MODELS_DIR = EVAL_DIR.parent / "models"
PYLIB_DIR = EVAL_DIR.parent / "pylib"

TESTSETS = {
    "ja_prose": "test_ja_prose.txt",
    "en_prose": "test_en_prose.txt",
    "code": "test_code.txt",
    "mixed_ja_code": "test_mixed_ja_code.txt",
}

# kototok models: name -> (file, tags). strict is the default profile of the v2 models.
KOTOTOK = {
    "kototok v2-T strict (200k)": ("kototok_v2s_200k.ktok", None),
    "kototok v2-T balanced (200k)": ("kototok_v2s_200k.ktok", ["ja", "casual", "kozh", "okuri"]),
    "kototok v2-BKO3 (200k)": ("kototok_v2_bko3_200k.ktok", None),
}

# Hugging Face `tokenizer.json` files in artifacts_real/external/.
HF_EXTERNAL = {
    "Inkling 2 Mini (201k)": "inkling2_tokenizer.json",
    "Qwen2.5 (152k)": "qwen25_tokenizer.json",
    "Llama 3 (128k)": "llama3_tokenizer.json",
    "DeepSeek V3 (128k)": "deepseek_tokenizer.json",
    "llm-jp (100k)": "llmjp_tokenizer.json",
}

TIKTOKEN = {"o200k_base (200k)": "o200k_base", "cl100k_base (100k)": "cl100k_base"}


def read_sets(which: list[str]) -> dict[str, str]:
    out = {}
    for name in which:
        path = DATA_DIR / TESTSETS[name]
        out[name] = path.read_text(encoding="utf-8")
    return out


def speed(fn, text: str, iters: int = 3) -> float:
    """Encode rate in MB/s of UTF-8 input, one warm-up run excluded."""
    size = len(text.encode("utf-8"))
    fn(text)
    best = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(text)
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, size / dt / 1e6)
    return round(best, 1)


def measure(name: str, encode, sets: dict[str, str], speed_set: str) -> dict:
    row = {"tokenizer": name, "sets": {}}
    for set_name, text in sets.items():
        ids = encode(text)
        row["sets"][set_name] = {
            "tokens": len(ids),
            "chars_per_token": round(len(text) / len(ids), 4) if ids else 0.0,
            "bytes_per_token": round(len(text.encode("utf-8")) / len(ids), 4) if ids else 0.0,
        }
    row["mb_per_s"] = speed(encode, sets[speed_set])
    return row


def kototok_encoder(file: str, tags):
    sys.path.insert(0, str(PYLIB_DIR))
    import kototok  # noqa: PLC0415

    tok = kototok.Tokenizer.load(str(MODELS_DIR / file), tags=tags)
    return lambda text: tok.encode(text)


def hf_encoder(file: str):
    from tokenizers import Tokenizer  # noqa: PLC0415

    tok = Tokenizer.from_file(str(EXTERNAL_DIR / file))
    return lambda text: tok.encode(text).ids


def tiktoken_encoder(name: str):
    import tiktoken  # noqa: PLC0415

    enc = tiktoken.get_encoding(name)
    return lambda text: enc.encode_ordinary(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated tokenizer names to keep")
    ap.add_argument("--sets", default=",".join(TESTSETS), help="comma-separated test sets")
    ap.add_argument("--speed-set", default="ja_prose")
    ap.add_argument("--json", default="", help="write the rows to this file")
    args = ap.parse_args()

    sets = read_sets([s for s in args.sets.split(",") if s])
    wanted = {m for m in args.models.split(",") if m}
    rows = []

    for name, (file, tags) in KOTOTOK.items():
        if wanted and name not in wanted:
            continue
        if not (MODELS_DIR / file).exists():
            print(f"[skip] {name}: {file} missing", file=sys.stderr)
            continue
        rows.append(measure(name, kototok_encoder(file, tags), sets, args.speed_set))

    for name, file in HF_EXTERNAL.items():
        if wanted and name not in wanted:
            continue
        if not (EXTERNAL_DIR / file).exists():
            print(f"[skip] {name}: {file} missing", file=sys.stderr)
            continue
        rows.append(measure(name, hf_encoder(file), sets, args.speed_set))

    for name, encoding in TIKTOKEN.items():
        if wanted and name not in wanted:
            continue
        try:
            enc = tiktoken_encoder(encoding)
        except Exception as exc:  # noqa: BLE001 - offline cache miss
            print(f"[skip] {name}: {exc}", file=sys.stderr)
            continue
        rows.append(measure(name, enc, sets, args.speed_set))

    width = max(len(r["tokenizer"]) for r in rows)
    header = f"{'tokenizer'.ljust(width)}  " + "  ".join(
        f"{s + ' c/t':>11} {s + ' B/t':>11}" for s in sets
    ) + f"  {'MB/s':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for s in sets:
            st = r["sets"][s]
            cells.append(f"{st['chars_per_token']:>11.4f} {st['bytes_per_token']:>11.4f}")
        print(f"{r['tokenizer'].ljust(width)}  " + "  ".join(cells) + f"  {r['mb_per_s']:>6.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
