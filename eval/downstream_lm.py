#!/usr/bin/env python3
"""Downstream task: train an identical tiny GPT on each tokenizer's output.

This is the real downstream check behind the trigram proxy: the same
jawiki bytes are tokenized by each candidate tokenizer, an identical
small decoder-only transformer is trained for one epoch on the resulting
token stream, and the held-out cross-entropy is reported normalized per
UTF-8 byte:

    bits/byte = -sum_t log2 p_theta(x_t | x_<t) / total_eval_bytes

Fairness notes:

* Same text for every tokenizer (default 48MiB jawiki slice whose first
  2MiB are skipped to stay disjoint from all eval sets — identical to
  ``downstream_trigram.py``). Better compression means more of the
  document fits inside a fixed context window; that advantage is the
  thing being measured, so the byte budget is what is fixed.
* Same architecture, width, depth, context, optimizer, and step budget
  in *tokens* — embedding size follows each tokenizer's vocab, which is
  itself a real cost of a large vocabulary.
* Eval is teacher-forced NLL over non-overlapping context windows; every
  model predicts the same raw bytes.

Usage (from eval/)::

    python downstream_lm.py --models 'v2s strict,sarashina2,o200k_base'
    python downstream_lm.py --device cuda --train-mb 64

Requires torch plus the eval venv tokenizer deps.
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


def train_slice(path: Path, skip: int, nbytes: int) -> list[str]:
    with open(path, "rb") as f:
        f.seek(skip)
        raw = f.read(nbytes)
    start = raw.find(b"\n") + 1
    end = raw.rfind(b"\n")
    text = raw[start:end].decode("utf-8", errors="replace")
    return [ln for ln in text.split("\n") if ln.strip()]


def read_eval(path: Path) -> tuple[list[str], int]:
    lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line)["text"] if line.startswith("{") else line)
    nbytes = sum(len(ln.encode("utf-8")) for ln in lines)
    return lines, nbytes


def build_gpt(vocab: int, ctx: int, d: int, layers: int, heads: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.proj = nn.Linear(d, d)
            self.heads = heads

        def forward(self, x):
            b, t, _ = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(b, t, self.heads, d // self.heads).transpose(1, 2)
            k = k.view(b, t, self.heads, d // self.heads).transpose(1, 2)
            v = v.view(b, t, self.heads, d // self.heads).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).contiguous().view(b, t, d))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        """Hidden-state GPT: logits are applied chunk-wise by the caller so
        peak activation stays ~chunk*vocab instead of batch*ctx*vocab."""

        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(ctx, d)
            self.blocks = nn.ModuleList(Block() for _ in range(layers))
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)
            self.head.weight = self.emb.weight
            self.apply(self._init)
            with torch.no_grad():
                self.emb.weight.normal_(0.0, 0.02)
                self.pos.weight.normal_(0.0, 0.02)

        @staticmethod
        def _init(m) -> None:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def hidden(self, idx):
            t = idx.shape[1]
            x = self.emb(idx) + self.pos(torch.arange(t, device=idx.device))
            for blk in self.blocks:
                x = blk(x)
            return self.ln(x)

    return GPT()


def chunked_ce(hidden, weight, targets, chunk: int = 1024, backprop: bool = True):
    """Cross-entropy of hidden@weight.T vs targets, computed in position
    chunks. With ``backprop`` each chunk's graph is freed right after its
    backward call, so peak logits memory is ~chunk*vocab, not N*vocab.
    Returns the mean loss over all positions."""
    import torch
    import torch.nn.functional as F

    total = targets.numel()
    loss_val = 0.0
    hs_list = list(hidden.split(chunk))
    ys_list = list(targets.split(chunk))
    for i, (hs, ys) in enumerate(zip(hs_list, ys_list)):
        l = F.cross_entropy(F.linear(hs, weight), ys, reduction="sum")
        if backprop:
            (l / total).backward(retain_graph=i < len(hs_list) - 1)
        loss_val += l.item()
    return loss_val / total


def train_one(
    label: str,
    enc: EncodeFn,
    vocab: int,
    lines: list[str],
    eval_ids: list[int],
    eval_bytes: int,
    args: argparse.Namespace,
) -> tuple[float, int, int, float]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    device = args.device
    ctx = args.ctx

    t0 = time.time()
    ids: list[int] = []
    for ln in lines:
        ids.extend(enc(ln))
        ids.append(vocab)  # EOS separator between documents
    n_train = len(ids)
    x = torch.tensor(ids, dtype=torch.long)
    n_win = (n_train - 1) // ctx
    # Fixed token budget: one pass over the corpus (capped by --steps).
    perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(0))
    steps = min(args.steps, n_win // args.batch)
    model = build_gpt(vocab + 1, ctx, args.d_model, args.layers, args.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()
    it = iter(perm)
    step = 0
    for step in range(steps):
        idx = torch.as_tensor([next(it) for _ in range(args.batch)])
        offs = (idx * ctx).unsqueeze(1) + torch.arange(ctx + 1)
        seq = x[offs].to(device)
        h = model.hidden(seq[:, :-1]).reshape(-1, args.d_model)
        opt.zero_grad(set_to_none=True)
        loss_val = chunked_ce(h, model.head.weight, seq[:, 1:].reshape(-1))
        opt.step()
        if step % 50 == 0:
            print(
                f"    [{label}] step {step}/{steps} loss {loss_val:.3f}",
                file=sys.stderr,
                flush=True,
            )
    # Teacher-forced eval over non-overlapping windows.
    model.eval()
    bits = 0.0
    with torch.no_grad():
        for i in range(0, len(eval_ids) - ctx, ctx):
            seq = torch.tensor(
                eval_ids[i : i + ctx + 1], dtype=torch.long, device=device
            ).unsqueeze(0)
            h = model.hidden(seq[:, :-1]).reshape(-1, args.d_model)
            nll_sum = 0.0
            for hs, ys in zip(
                h.split(1024), seq[:, 1:].reshape(-1).split(1024)
            ):
                nll_sum += F.cross_entropy(
                    F.linear(hs, model.head.weight), ys, reduction="sum"
                ).item()
            bits += nll_sum / math.log(2.0)
    bpb = bits / eval_bytes
    elapsed = time.time() - t0
    return bpb, n_train, n_params, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train_corpus_jw.txt")
    parser.add_argument("--eval", type=Path, default=DATA_DIR / "test_ja_prose.txt")
    parser.add_argument("--train-mb", type=float, default=48.0)
    parser.add_argument("--train-skip-mb", type=float, default=2.0)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--models", type=str, default="*")
    parser.add_argument("--pylib", type=Path, default=REPO_ROOT / "pylib")
    args = parser.parse_args()

    import fnmatch

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        torch.set_num_threads(max(1, (torch.get_num_threads() or 16)))
    print(f"[env] device={args.device}", file=sys.stderr)

    lines = train_slice(
        args.train, int(args.train_skip_mb * 2**20), int(args.train_mb * 2**20)
    )
    eval_lines, eval_bytes = read_eval(args.eval)
    print(
        f"[data] train {len(lines)} lines, eval {eval_bytes / 2**10:.0f}KiB/"
        f"{len(eval_lines)} lines",
        file=sys.stderr,
        flush=True,
    )

    lineup: list[tuple[str, EncodeFn, int]] = []
    for label, fname, tags in KOTOTOK_MODELS:
        lineup.append((label, *kt_enc(MODELS_DIR / fname, tags, args.pylib)))
    for name in EXTERNAL_SP:
        lineup.append((name, *sp_enc(EXTERNAL_DIR / EXTERNAL_FILES[name])))
    for name in EXTERNAL_HF:
        lineup.append((name, *hf_enc(EXTERNAL_DIR / EXTERNAL_FILES[name])))
    for name in EXTERNAL_TIK:
        lineup.append((name, *tik_enc(name)))

    pats = [p.strip() for p in args.models.split(",")]
    results: list[tuple[str, float, int, int, float]] = []
    for label, enc, vocab in lineup:
        if not any(fnmatch.fnmatch(label, p) for p in pats):
            continue
        eval_ids: list[int] = []
        for ln in eval_lines:
            eval_ids.extend(enc(ln))
            eval_ids.append(vocab)
        try:
            bpb, n_train, n_params, dt = train_one(
                label, enc, vocab, lines, eval_ids, eval_bytes, args
            )
            results.append((label, bpb, n_train, n_params, dt))
            print(
                f"{label:14s} {bpb:.4f} bits/byte "
                f"({n_train / 1e6:.1f}M train tok, {n_params / 1e6:.1f}M params, "
                f"{dt / 60:.1f}min)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - keep the table going
            print(f"{label:14s} ERROR {e}", flush=True)

    print("\n=== sorted ===")
    for label, bpb, n_train, n_params, dt in sorted(results, key=lambda r: r[1]):
        print(
            f"{label:14s} {bpb:.4f} bits/byte  "
            f"({n_train / 1e6:.1f}M tok, {n_params / 1e6:.1f}M params, "
            f"{dt / 60:.1f}min)"
        )


if __name__ == "__main__":
    main()
