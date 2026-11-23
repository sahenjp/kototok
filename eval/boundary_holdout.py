#!/usr/bin/env python3
"""Boundary metrics for arbitrary tokenizers against arbitrary gold JSONL sets.

Evaluates the same P/R/F1, okurigana-integrity, and noun+particle over-merge
metrics as ``run_eval.compute_boundary_metrics`` for a fixed model lineup —
used to produce the README "Matched held-out comparison" and
"Independent-gold cross-check" tables.

Gold files use the shared schema::

    {"text": str, "byte_spans": [[start, end], ...],
     "okurigana_words": [{"text", "byte_span"}],
     "noun_particles": [{"boundary_byte", ...}]}

Usage (from eval/)::

    python boundary_holdout.py \
        --gold sudachi=artifacts_real/data/gold_sudachi_holdout.jsonl \
        --gold unidic=artifacts_real/data/gold_unidic_holdout.jsonl

Requires ``pylib/kototok.so`` on the default path (override with
``--pylib``), plus the eval venv deps (sentencepiece, tokenizers, tiktoken).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
MODELS_DIR = EVAL_DIR / "artifacts_real" / "models"
EXTERNAL_DIR = EVAL_DIR / "artifacts_real" / "external"

# kototok model lineup: (label, filename, tags)
KOTOTOK_MODELS: list[tuple[str, str, list[str] | None]] = [
    ("v2s strict", "kototok_v2s_200k.ktok", None),
    ("v2s balanced", "kototok_v2s_200k.ktok", ["ja", "casual", "kozh", "okuri"]),
    ("v2_bko3", "kototok_v2_bko3_200k.ktok", None),
    ("s3n_bko5", "kototok_s3n_bko5_200k.ktok", None),
    ("s3o_bko2", "kototok_s3o_bko2_200k.ktok", None),
]

EXTERNAL_SP = ["sarashina2", "plamo", "gemma2", "mistral"]
EXTERNAL_HF = ["llm-jp", "llama3", "qwen2.5", "deepseek"]
EXTERNAL_TIK = ["o200k_base", "cl100k_base"]

EXTERNAL_FILES = {
    "sarashina2": "sarashina2_tokenizer.model",
    "plamo": "plamo_tokenizer.model",
    "gemma2": "gemma2_tokenizer.model",
    "mistral": "mistral_tokenizer.model",
    "llm-jp": "llmjp_tokenizer.json",
    "llama3": "llama3_tokenizer.json",
    "qwen2.5": "qwen25_tokenizer.json",
    "deepseek": "deepseek_tokenizer.json",
}

ByteLensFn = Callable[[str], list[int]]


def kt_lens(path: Path, tags: list[str] | None, pylib: Path) -> ByteLensFn:
    sys.path.insert(0, str(pylib))
    import kototok  # noqa: PLC0415

    tok = kototok.Tokenizer.load(str(path), tags=tags)

    def f(text: str) -> list[int]:
        return [end - start for _id, start, end in tok.encode_with_offsets(text)]

    return f


def hf_lens(path: Path) -> ByteLensFn:
    from tokenizers import Tokenizer  # noqa: PLC0415

    tok = Tokenizer.from_file(str(path))

    def f(text: str) -> list[int]:
        return [end - start for start, end in tok.encode(text).offsets]

    return f


def sp_lens(path: Path) -> ByteLensFn:
    import sentencepiece as spm  # noqa: PLC0415

    proc = spm.SentencePieceProcessor(model_file=str(path))

    def f(text: str) -> list[int]:
        return [
            len(piece.replace("▁", " ").encode("utf-8"))
            for piece in proc.encode(text, out_type=str)
        ]

    return f


def tik_lens(enc_name: str) -> ByteLensFn:
    import tiktoken  # noqa: PLC0415

    enc = tiktoken.get_encoding(enc_name)

    def f(text: str) -> list[int]:
        return [
            len(enc.decode_single_token_bytes(tid))
            for tid in enc.encode(text, disallowed_special=())
        ]

    return f


def boundary_metrics(
    gold_rows: list[dict[str, Any]], byte_lens_fn: ByteLensFn
) -> dict[str, float]:
    tp = pred_total = gold_total = 0
    oku_total = oku_intact = np_total = np_overmerged = 0

    for row in gold_rows:
        text_bytes = row["text"].encode("utf-8")
        lens = byte_lens_fn(row["text"])
        pred: set[int] = set()
        pos = 0
        for bl in lens[:-1]:
            pos += bl
            pred.add(pos)
        pred = {b for b in pred if 0 < b < len(text_bytes)}

        gold = {
            span[1] for span in row["byte_spans"][:-1] if 0 < span[1] < len(text_bytes)
        }
        tp += len(pred & gold)
        pred_total += len(pred)
        gold_total += len(gold)

        for oku in row["okurigana_words"]:
            a, b = oku["byte_span"]
            oku_total += 1
            if not any(a < p < b for p in pred):
                oku_intact += 1
        for np_item in row["noun_particles"]:
            np_total += 1
            if np_item["boundary_byte"] not in pred:
                np_overmerged += 1

    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "okurigana_integrity_pct": round(oku_intact / oku_total * 100, 1)
        if oku_total
        else 0.0,
        "noun_particle_overmerge_pct": round(np_overmerged / np_total * 100, 1)
        if np_total
        else 0.0,
    }


def build_lineup(pylib: Path) -> list[tuple[str, ByteLensFn]]:
    lineup: list[tuple[str, ByteLensFn]] = []
    for label, fname, tags in KOTOTOK_MODELS:
        lineup.append((label, kt_lens(MODELS_DIR / fname, tags, pylib)))
    for name in EXTERNAL_SP:
        lineup.append((name, sp_lens(EXTERNAL_DIR / EXTERNAL_FILES[name])))
    for name in EXTERNAL_HF:
        lineup.append((name, hf_lens(EXTERNAL_DIR / EXTERNAL_FILES[name])))
    for name in EXTERNAL_TIK:
        lineup.append((name, tik_lens(name)))
    return lineup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boundary metrics for kototok + external tokenizers vs gold JSONL.",
    )
    parser.add_argument(
        "--gold",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Gold JSONL; repeatable. NAME labels the report section.",
    )
    parser.add_argument(
        "--pylib",
        type=Path,
        default=REPO_ROOT / "pylib",
        help="Directory containing kototok.so (default: repo pylib/).",
    )
    args = parser.parse_args()

    gold_sets: list[tuple[str, list[dict[str, Any]]]] = []
    for spec in args.gold:
        name, _, path = spec.partition("=")
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        gold_sets.append((name, rows))

    lineup = build_lineup(args.pylib)

    for gold_name, rows in gold_sets:
        print(f"=== {gold_name} gold ({len(rows)} lines) ===")
        for label, fn in lineup:
            try:
                m = boundary_metrics(rows, fn)
                print(
                    f"{label:14s} F1={m['f1']:.4f} P={m['precision']:.4f} "
                    f"R={m['recall']:.4f} oku={m['okurigana_integrity_pct']:.1f}% "
                    f"np_over={m['noun_particle_overmerge_pct']:.1f}%",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 - keep the table going
                print(f"{label:14s} ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
