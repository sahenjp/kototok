#!/usr/bin/env python3
"""Downstream proxy: trigram-LM cross-entropy (bits/byte, jawiki held-out).

For each tokenizer, a trigram language model is trained on a fixed jawiki
training slice tokenized by *that* tokenizer, then the cross-entropy of a
held-out jawiki text is measured and normalized per UTF-8 byte:

    bits/byte = -sum_i log2 P(t_i | t_{i-2}, t_{i-1}) / total_eval_bytes

This jointly measures compression (fewer tokens for the same bytes helps)
and predictability (tokens that form learnable n-gram regularities help),
which is why it is the cheapest honest proxy for downstream LM quality.

Smoothing: recursive absolute discounting with a single discount D
(Kneser-Ney without per-count discount tiers):

    P(c | a,b) = max(n3-D, 0)/N(a,b) + D*f(a,b)/N(a,b) * P(c | b)
    P(c | b)   = max(n2-D, 0)/N(b)   + D*f(b)/N(b)   * P_uni(c)
    P_uni(c)   = continuation(c)/S   (or a fixed floor if never preceded)

where f(.) is the number of distinct followers of the context and
continuation(c) counts the distinct tokens seen immediately before c —
the standard KN lower-order distribution. All counts come from the
tokenized training slice only; every tokenizer sees identical raw text.

The training slice defaults to bytes [2MiB, 2MiB+64MiB) of
``train_corpus_jw.txt``: the file opens with jawiki prose and the
500-line boundary holdout lives entirely inside the first ~1.6MiB, so
the offset guarantees zero overlap between LM training text and any
eval set. The eval text defaults to ``test_ja_prose.txt`` (jawiki
articles whose titles were never fetched for any training corpus).

Usage (from eval/)::

    python downstream_trigram.py                      # full lineup
    python downstream_trigram.py --models 'v2s*,s3o*' # kototok only
    python downstream_trigram.py --train-mb 32 --eval artifacts_real/data/test_ja_prose.txt

Requires ``pylib/kototok.so`` plus the eval venv deps (sentencepiece,
tokenizers, tiktoken).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
MODELS_DIR = EVAL_DIR / "artifacts_real" / "models"
EXTERNAL_DIR = EVAL_DIR / "artifacts_real" / "external"
DATA_DIR = EVAL_DIR / "artifacts_real" / "data"

# kototok model lineup: (label, filename, tags)
KOTOTOK_MODELS: list[tuple[str, str, list[str] | None]] = [
    ("v2s strict", "kototok_v2s_200k.ktok", None),
    ("v2s balanced", "kototok_v2s_200k.ktok", ["ja", "casual", "kozh", "okuri"]),
    ("v2_bko3", "kototok_v2_bko3_200k.ktok", None),
    ("s3n_bko5", "kototok_s3n_bko5_200k.ktok", None),
    ("s3o_bko2", "kototok_s3o_bko2_200k.ktok", None),
    ("s3o_bko3", "kototok_s3o_bko3_200k.ktok", None),
]

EXTERNAL_SP = ["sarashina2", "plamo", "gemma2", "mistral"]
EXTERNAL_HF = ["llm-jp", "llama3", "qwen2.5", "deepseek", "inkling2"]
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
    "inkling2": "inkling2_tokenizer.json",
}

EncodeFn = Callable[[str], list[int]]


def kt_enc(path: Path, tags: list[str] | None, pylib: Path) -> tuple[EncodeFn, int]:
    sys.path.insert(0, str(pylib))
    import kototok  # noqa: PLC0415

    tok = kototok.Tokenizer.load(str(path), tags=tags)
    return tok.encode, tok.vocab_size


def hf_enc(path: Path) -> tuple[EncodeFn, int]:
    from tokenizers import Tokenizer  # noqa: PLC0415

    tok = Tokenizer.from_file(str(path))
    vocab = tok.get_vocab_size()
    return lambda t: tok.encode(t).ids, vocab


def sp_enc(path: Path) -> tuple[EncodeFn, int]:
    import sentencepiece as spm  # noqa: PLC0415

    proc = spm.SentencePieceProcessor(model_file=str(path))
    return proc.encode, proc.get_piece_size()


def tik_enc(enc_name: str) -> tuple[EncodeFn, int]:
    import tiktoken  # noqa: PLC0415

    enc = tiktoken.get_encoding(enc_name)
    return lambda t: enc.encode(t, disallowed_special=()), enc.n_vocab


def read_lines(path: Path) -> list[str]:
    """Eval lines: JSONL with a 'text' field, or a plain-text file."""
    lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                lines.append(json.loads(line)["text"])
            else:
                lines.append(line)
    return lines


def train_slice(path: Path, skip: int, nbytes: int) -> str:
    with open(path, "rb") as f:
        f.seek(skip)
        raw = f.read(nbytes)
    # Drop the first (possibly mid-UTF-8) partial line and the tail stub.
    start = raw.find(b"\n") + 1
    end = raw.rfind(b"\n")
    return raw[start:end].decode("utf-8", errors="replace")


class TrigramLM:
    """Single-discount absolute-discounting trigram with KN continuation base."""

    def __init__(self, vocab: int, discount: float) -> None:
        self.d = discount
        self.bos = vocab
        self.eos = vocab + 1
        self.uni: dict[int, int] = {}
        self.bi: dict[int, dict[int, int]] = {}
        self.tri: dict[tuple[int, int], dict[int, int]] = {}
        self.rev: dict[int, int] = {}
        self.s_cont = 0
        self.n_train = 0

    def observe(self, ids: list[int]) -> None:
        seq = [self.bos, self.bos, *ids, self.eos]
        uni, bi, tri = self.uni, self.bi, self.tri
        a, b = seq[0], seq[1]
        for c in seq[2:]:
            uni[c] = uni.get(c, 0) + 1
            ctx_b = bi.get(b)
            if ctx_b is None:
                bi[b] = {c: 1}
            else:
                ctx_b[c] = ctx_b.get(c, 0) + 1
            ctx_ab = tri.get((a, b))
            if ctx_ab is None:
                tri[(a, b)] = {c: 1}
            else:
                ctx_ab[c] = ctx_ab.get(c, 0) + 1
            a, b = b, c
        self.n_train += len(seq) - 2

    def finalize(self) -> None:
        rev: dict[int, int] = {}
        for followers in self.bi.values():
            for c in followers:
                rev[c] = rev.get(c, 0) + 1
        self.rev = rev
        self.s_cont = sum(rev.values()) or 1

    def p_uni(self, c: int) -> float:
        # KN continuation count; 0.1 floor keeps unseen followers finite.
        return (self.rev.get(c, 0) + 0.1) / (self.s_cont + 0.1)

    def p_bi(self, b: int, c: int) -> float:
        foll = self.bi.get(b)
        if not foll:
            return self.p_uni(c)
        tot = sum(foll.values())
        return max(foll.get(c, 0) - self.d, 0.0) / tot + (
            self.d * len(foll) / tot
        ) * self.p_uni(c)

    def p_tri(self, a: int, b: int, c: int) -> float:
        foll = self.tri.get((a, b))
        if not foll:
            return self.p_bi(b, c)
        tot = sum(foll.values())
        return max(foll.get(c, 0) - self.d, 0.0) / tot + (
            self.d * len(foll) / tot
        ) * self.p_bi(b, c)

    def cross_entropy_bits(self, ids: list[int]) -> tuple[float, int]:
        seq = [self.bos, self.bos, *ids, self.eos]
        bits = 0.0
        a, b = seq[0], seq[1]
        for c in seq[2:]:
            bits -= math.log2(self.p_tri(a, b, c))
            a, b = b, c
        return bits, len(ids)


def build_lineup(pylib: Path) -> list[tuple[str, EncodeFn, int]]:
    lineup: list[tuple[str, EncodeFn, int]] = []
    for label, fname, tags in KOTOTOK_MODELS:
        enc, vocab = kt_enc(MODELS_DIR / fname, tags, pylib)
        lineup.append((label, enc, vocab))
    for name in EXTERNAL_SP:
        enc, vocab = sp_enc(EXTERNAL_DIR / EXTERNAL_FILES[name])
        lineup.append((name, enc, vocab))
    for name in EXTERNAL_HF:
        enc, vocab = hf_enc(EXTERNAL_DIR / EXTERNAL_FILES[name])
        lineup.append((name, enc, vocab))
    for name in EXTERNAL_TIK:
        enc, vocab = tik_enc(name)
        lineup.append((name, enc, vocab))
    return lineup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigram-LM cross-entropy (bits/byte) for kototok + externals.",
    )
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train_corpus_jw.txt")
    parser.add_argument("--eval", type=Path, default=DATA_DIR / "test_ja_prose.txt")
    parser.add_argument("--train-mb", type=float, default=64.0)
    parser.add_argument("--train-skip-mb", type=float, default=2.0)
    parser.add_argument("--discount", type=float, default=0.75)
    parser.add_argument(
        "--models",
        type=str,
        default="*",
        help="Comma-separated fnmatch patterns over model labels.",
    )
    parser.add_argument("--pylib", type=Path, default=REPO_ROOT / "pylib")
    args = parser.parse_args()

    import fnmatch

    pats = [p.strip() for p in args.models.split(",")]

    train_text = train_slice(
        args.train, int(args.train_skip_mb * 2**20), int(args.train_mb * 2**20)
    )
    train_lines = [ln for ln in train_text.split("\n") if ln.strip()]
    eval_lines = read_lines(args.eval)
    eval_bytes = sum(len(ln.encode("utf-8")) for ln in eval_lines)
    print(
        f"[data] train {len(train_text.encode('utf-8')) / 2**20:.1f}MiB/"
        f"{len(train_lines)} lines, eval {eval_bytes / 2**10:.0f}KiB/"
        f"{len(eval_lines)} lines ({args.eval.name})",
        file=sys.stderr,
        flush=True,
    )

    results: list[tuple[str, float, int, int]] = []
    for label, enc, vocab in build_lineup(args.pylib):
        if not any(fnmatch.fnmatch(label, p) for p in pats):
            continue
        t0 = time.time()
        lm = TrigramLM(vocab, args.discount)
        try:
            for ln in train_lines:
                lm.observe(enc(ln))
            lm.finalize()
            bits = tokens = 0
            for ln in eval_lines:
                b, n = lm.cross_entropy_bits(enc(ln))
                bits += b
                tokens += n
            bpb = bits / eval_bytes
            results.append((label, bpb, tokens, lm.n_train))
            print(
                f"{label:14s} {bpb:.4f} bits/byte  "
                f"(eval {tokens} tok, train {lm.n_train / 1e6:.1f}M tok, "
                f"{time.time() - t0:.0f}s)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - keep the table going
            print(f"{label:14s} ERROR {e}", flush=True)

    print("\n=== sorted ===")
    for label, bpb, tokens, _ in sorted(results, key=lambda r: r[1]):
        print(f"{label:14s} {bpb:.4f} bits/byte  ({tokens} eval tokens)")


if __name__ == "__main__":
    main()
