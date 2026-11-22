#!/usr/bin/env python3
"""Reproducible evaluation suite for kototok (AsymScript-BPE) and baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

KOTOTOK_PRESETS = [
    "asym-full",
    "asym-no-phase2",
    "asym-no-picky",
    "asym-no-acsg",
    "strict-script",
    "standard-bbpe",
]

TEST_SPLITS = {
    "ja_prose": "test_ja_prose.txt",
    "en_prose": "test_en_prose.txt",
    "code": "test_code.txt",
    "mixed_ja_code": "test_mixed_ja_code.txt",
}


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Return GPT-2 ByteLevel byte -> unicode char mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


GPT2_UNICODE_TO_BYTE: dict[str, int] = {v: k for k, v in gpt2_bytes_to_unicode().items()}


def decode_bytelevel_token_to_bytes(tok: str) -> bytes | None:
    """Decode a ByteLevel BPE token string into its raw byte sequence."""
    out = bytearray()
    for ch in tok:
        b = GPT2_UNICODE_TO_BYTE.get(ch)
        if b is None:
            return None
        out.append(b)
    return bytes(out)


def is_valid_utf8_bytes(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def compute_boundary_metrics(
    gold_rows: list[dict[str, Any]],
    encode_to_byte_lens: Any,
) -> dict[str, float]:
    """Compute Boundary P/R/F1, Okurigana Integrity Rate, and Noun+Particle Over-Merge Rate."""
    tp = 0
    pred_total = 0
    gold_total = 0
    oku_total = 0
    oku_intact = 0
    np_total = 0
    np_overmerged = 0

    for row in gold_rows:
        text: str = row["text"]
        text_bytes = text.encode("utf-8")
        byte_lens: list[int] = encode_to_byte_lens(text)

        # Build predicted internal boundary byte offsets
        pred_boundaries: set[int] = set()
        pos = 0
        for bl in byte_lens[:-1]:
            pos += bl
            pred_boundaries.add(pos)
        if byte_lens:
            pos += byte_lens[-1]
        if pos != len(text_bytes):
            # Clamp/filter to valid text length if any mismatch
            pred_boundaries = {b for b in pred_boundaries if 0 < b < len(text_bytes)}

        gold_boundaries = {
            span[1] for span in row["byte_spans"][:-1] if 0 < span[1] < len(text_bytes)
        }

        tp += len(pred_boundaries & gold_boundaries)
        pred_total += len(pred_boundaries)
        gold_total += len(gold_boundaries)

        for oku in row["okurigana_words"]:
            a, b = oku["byte_span"]
            oku_total += 1
            # Intact iff no predicted boundary falls strictly inside (a, b)
            if not any(a < p < b for p in pred_boundaries):
                oku_intact += 1

        for np_item in row["noun_particles"]:
            boundary_b = np_item["boundary_byte"]
            np_total += 1
            if boundary_b not in pred_boundaries:
                np_overmerged += 1

    precision = (tp / pred_total) if pred_total else 0.0
    recall = (tp / gold_total) if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    oku_rate = (oku_intact / oku_total * 100.0) if oku_total else 0.0
    np_overmerge_rate = (np_overmerged / np_total * 100.0) if np_total else 0.0

    return {
        "boundary_precision": round(precision, 4),
        "boundary_recall": round(recall, 4),
        "boundary_f1": round(f1, 4),
        "okurigana_integrity_pct": round(oku_rate, 2),
        "noun_particle_overmerge_pct": round(np_overmerge_rate, 2),
        "okurigana_intact_count": oku_intact,
        "okurigana_total_count": oku_total,
        "noun_particle_overmerge_count": np_overmerged,
        "noun_particle_total_count": np_total,
    }


def align_lossy_pieces_to_bytes(text: str, pieces: list[str]) -> list[int]:
    """Recover per-piece byte lengths even if `pieces` contains U+FFFD from partial UTF-8."""
    if "".join(pieces) == text:
        return [len(p.encode("utf-8")) for p in pieces]

    raw = text.encode("utf-8")
    n = len(raw)
    pos = 0
    lens: list[int] = []
    for piece in pieces:
        start = pos
        for ch in piece:
            if ch != "\ufffd":
                ch_bytes = ch.encode("utf-8")
                idx = raw.find(ch_bytes, pos)
                if idx != -1:
                    pos = idx + len(ch_bytes)
                else:
                    pos = min(n, pos + len(ch_bytes))
            else:
                # Advance by at least 1 byte for a replacement character
                pos = min(n, pos + 1)
        lens.append(max(1, pos - start) if start < n else 0)
    if sum(lens) != n and lens:
        lens[-1] += n - sum(lens)
    return lens


def compression_stats(text: str, token_count: int) -> dict[str, Any]:
    byte_count = len(text.encode("utf-8"))
    char_count = len(text)
    return {
        "tokens": token_count,
        "chars": char_count,
        "bytes": byte_count,
        "chars_per_token": round(char_count / token_count, 4) if token_count else 0.0,
        "bytes_per_token": round(byte_count / token_count, 4) if token_count else 0.0,
    }


def bench_python_fn(fn: Any, text: str, iters: int = 5) -> dict[str, float]:
    """Benchmark a Python encode function with 1 warmup and `iters` timed runs."""
    _ = fn(text)  # warmup
    t0 = time.perf_counter()
    toks = 0
    for _ in range(iters):
        res = fn(text)
        toks = len(res)
    elapsed = time.perf_counter() - t0
    byte_len = len(text.encode("utf-8"))
    avg_secs = elapsed / iters
    mb_per_sec = (byte_len / 1_000_000.0) / avg_secs if avg_secs > 0 else 0.0
    return {
        "iters": iters,
        "tokens": toks,
        "total_secs": round(elapsed, 4),
        "avg_ms": round(avg_secs * 1000.0, 2),
        "mb_per_sec": round(mb_per_sec, 2),
    }


# ---------------------------------------------------------------------------
# HuggingFace & tiktoken baseline evaluators
# ---------------------------------------------------------------------------

def train_hf_bpe(train_path: Path, vocab_size: int, save_path: Path) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([str(train_path)], trainer)
    tok.save(str(save_path))
    return tok


def train_hf_unigram(train_path: Path, vocab_size: int, save_path: Path) -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split("\n", behavior="isolated"),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Digits(individual_digits=True),
        ]
    )
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        unk_token="<unk>",
        shrinking_factor=0.75,
        n_sub_iterations=2,
    )
    tok.train([str(train_path)], trainer)
    tok.save(str(save_path))
    return tok


def evaluate_hf_bpe(
    tok: Tokenizer,
    test_texts: dict[str, str],
    adv_text: str,
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    vocab = tok.get_vocab()
    vocab_size = len(vocab)
    learned_tokens = 0
    invalid_learned = 0
    for piece in vocab:
        raw = decode_bytelevel_token_to_bytes(piece)
        if raw is None or len(raw) < 2:
            continue
        learned_tokens += 1
        if not is_valid_utf8_bytes(raw):
            invalid_learned += 1

    comp: dict[str, Any] = {}
    used_ids: set[int] = set()
    for name, txt in test_texts.items():
        enc = tok.encode(txt)
        used_ids.update(enc.ids)
        comp[name] = compression_stats(txt, len(enc.ids))

    def byte_lens_fn(s: str) -> list[int]:
        enc = tok.encode(s)
        lens: list[int] = []
        for t in enc.tokens:
            raw = decode_bytelevel_token_to_bytes(t)
            lens.append(len(raw) if raw is not None else len(t.encode("utf-8")))
        return lens

    ja_bench = bench_python_fn(lambda s: tok.encode(s).ids, test_texts["ja_prose"], iters=5)
    adv_bench = bench_python_fn(lambda s: tok.encode(s).ids, adv_text, iters=3)
    dead_rate = (vocab_size - len(used_ids)) / vocab_size * 100.0 if vocab_size else 0.0

    return {
        "label": "hf-bpe",
        "kind": "baseline",
        "vocab_size": vocab_size,
        "compression": comp,
        "utf8_audit": {
            "alive_non_fallback_tokens": learned_tokens,
            "invalid_utf8_learned_tokens": invalid_learned,
            "invalid_utf8_rate_pct": round((invalid_learned / learned_tokens * 100.0) if learned_tokens else 0.0, 2),
        },
        "ja_boundary": compute_boundary_metrics(gold_rows, byte_lens_fn),
        "dead_token_rate_pct": round(dead_rate, 2),
        "used_tokens_on_test": len(used_ids),
        "ja_throughput_mb_s": ja_bench["mb_per_sec"],
        "adversarial_1mb_ms": adv_bench["avg_ms"],
        "adversarial_1mb_mb_s": adv_bench["mb_per_sec"],
    }


def evaluate_hf_unigram(
    tok: Tokenizer,
    test_texts: dict[str, str],
    adv_text: str,
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    vocab = tok.get_vocab()
    vocab_size = len(vocab)
    non_special = [k for k in vocab if k != "<unk>"]

    comp: dict[str, Any] = {}
    used_ids: set[int] = set()
    for name, txt in test_texts.items():
        enc = tok.encode(txt)
        used_ids.update(enc.ids)
        comp[name] = compression_stats(txt, len(enc.ids))

    def byte_lens_fn(s: str) -> list[int]:
        enc = tok.encode(s)
        # Use exact character offsets mapped to UTF-8 byte lengths
        return [len(s[start:end].encode("utf-8")) for start, end in enc.offsets]

    ja_bench = bench_python_fn(lambda s: tok.encode(s).ids, test_texts["ja_prose"], iters=5)
    adv_bench = bench_python_fn(lambda s: tok.encode(s).ids, adv_text, iters=3)
    dead_rate = (vocab_size - len(used_ids)) / vocab_size * 100.0 if vocab_size else 0.0

    return {
        "label": "hf-unigram",
        "kind": "baseline",
        "vocab_size": vocab_size,
        "compression": comp,
        "utf8_audit": {
            "alive_non_fallback_tokens": len(non_special),
            "invalid_utf8_learned_tokens": 0,
            "invalid_utf8_rate_pct": 0.0,
        },
        "ja_boundary": compute_boundary_metrics(gold_rows, byte_lens_fn),
        "dead_token_rate_pct": round(dead_rate, 2),
        "used_tokens_on_test": len(used_ids),
        "ja_throughput_mb_s": ja_bench["mb_per_sec"],
        "adversarial_1mb_ms": adv_bench["avg_ms"],
        "adversarial_1mb_mb_s": adv_bench["mb_per_sec"],
    }


def evaluate_tiktoken(
    enc_name: str,
    display_label: str,
    test_texts: dict[str, str],
    adv_text: str,
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    enc = tiktoken.get_encoding(enc_name)
    vocab_size = enc.n_vocab
    learned_tokens = 0
    invalid_learned = 0
    for raw in enc.token_byte_values():
        if len(raw) < 2:
            continue
        learned_tokens += 1
        if not is_valid_utf8_bytes(raw):
            invalid_learned += 1

    comp: dict[str, Any] = {}
    used_ids: set[int] = set()
    for name, txt in test_texts.items():
        ids = enc.encode(txt, disallowed_special=())
        used_ids.update(ids)
        comp[name] = compression_stats(txt, len(ids))

    def byte_lens_fn(s: str) -> list[int]:
        ids = enc.encode(s, disallowed_special=())
        return [len(enc.decode_single_token_bytes(tid)) for tid in ids]

    ja_bench = bench_python_fn(
        lambda s: enc.encode(s, disallowed_special=()), test_texts["ja_prose"], iters=5
    )
    adv_bench = bench_python_fn(
        lambda s: enc.encode(s, disallowed_special=()), adv_text, iters=3
    )
    dead_rate = (vocab_size - len(used_ids)) / vocab_size * 100.0 if vocab_size else 0.0

    return {
        "label": display_label,
        "kind": "external",
        "vocab_size": vocab_size,
        "compression": comp,
        "utf8_audit": {
            "alive_non_fallback_tokens": learned_tokens,
            "invalid_utf8_learned_tokens": invalid_learned,
            "invalid_utf8_rate_pct": round((invalid_learned / learned_tokens * 100.0) if learned_tokens else 0.0, 2),
        },
        "ja_boundary": compute_boundary_metrics(gold_rows, byte_lens_fn),
        "dead_token_rate_pct": round(dead_rate, 2),
        "used_tokens_on_test": len(used_ids),
        "ja_throughput_mb_s": ja_bench["mb_per_sec"],
        "adversarial_1mb_ms": adv_bench["avg_ms"],
        "adversarial_1mb_mb_s": adv_bench["mb_per_sec"],
    }


# ---------------------------------------------------------------------------
# kototok CLI evaluator
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def evaluate_kototok_preset(
    bin_path: Path,
    preset: str,
    train_path: Path,
    model_path: Path,
    vocab_size: int,
    test_paths: dict[str, Path],
    test_texts: dict[str, str],
    adv_path: Path,
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_cmd([
        str(bin_path),
        "train",
        "--input", str(train_path),
        "--output", str(model_path),
        "--vocab-size", str(vocab_size),
        "--preset", preset,
    ])

    inspect_raw = run_cmd([
        str(bin_path),
        "inspect",
        "--model", str(model_path),
        "--json",
    ])
    inspect_json = json.loads(inspect_raw)

    comp: dict[str, Any] = {}
    used_ids: set[int] = set()
    for name, path in test_paths.items():
        enc_raw = run_cmd([
            str(bin_path),
            "encode",
            "--model", str(model_path),
            "--input", str(path),
            "--json",
        ])
        enc_json = json.loads(enc_raw)
        used_ids.update(enc_json["ids"])
        comp[name] = compression_stats(test_texts[name], enc_json["token_count"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / "sent.txt"

        def byte_lens_fn(s: str) -> list[int]:
            tmp_input.write_text(s, encoding="utf-8")
            out = json.loads(
                run_cmd([
                    str(bin_path),
                    "encode",
                    "--model", str(model_path),
                    "--input", str(tmp_input),
                    "--json",
                ])
            )
            if "byte_lens" in out:
                return [int(x) for x in out["byte_lens"]]
            return align_lossy_pieces_to_bytes(s, out["pieces"])

        ja_boundary = compute_boundary_metrics(gold_rows, byte_lens_fn)

    ja_bench = json.loads(
        run_cmd([
            str(bin_path),
            "bench",
            "--model", str(model_path),
            "--input", str(test_paths["ja_prose"]),
            "--threads", "1",
            "--iters", "5",
            "--json",
        ])
    )
    adv_bench = json.loads(
        run_cmd([
            str(bin_path),
            "bench",
            "--model", str(model_path),
            "--input", str(adv_path),
            "--threads", "1",
            "--iters", "5",
            "--json",
        ])
    )

    alive_vocab = inspect_json["alive_vocab_size"]
    dead_rate = (alive_vocab - len(used_ids)) / alive_vocab * 100.0 if alive_vocab else 0.0
    adv_iters = max(1, adv_bench.get("iters", 5))
    adv_ms = adv_bench["total_secs"] / adv_iters * 1000.0

    return {
        "label": f"kototok:{preset}",
        "kind": "kototok",
        "preset": preset,
        "vocab_size": alive_vocab,
        "compression": comp,
        "utf8_audit": {
            "alive_vocab_size": alive_vocab,
            "alive_non_fallback_tokens": inspect_json["alive_non_fallback_tokens"],
            "byte_fallback_tokens": inspect_json["byte_fallback_tokens"],
            "pruned_tokens": inspect_json["pruned_tokens"],
            "phase1_merged_alive": inspect_json["phase1_merged_alive"],
            "phase2_merged_alive": inspect_json["phase2_merged_alive"],
            "invalid_utf8_learned_tokens": inspect_json["invalid_utf8_learned_tokens"],
            "invalid_utf8_rate_pct": round(inspect_json["invalid_utf8_rate"] * 100.0, 2),
        },
        "ja_boundary": ja_boundary,
        "dead_token_rate_pct": round(dead_rate, 2),
        "used_tokens_on_test": len(used_ids),
        "ja_throughput_mb_s": round(ja_bench["mb_per_sec"], 2),
        "adversarial_1mb_ms": round(adv_ms, 2),
        "adversarial_1mb_mb_s": round(adv_bench["mb_per_sec"], 2),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def render_markdown_report(results: list[dict[str, Any]], vocab_size: int) -> str:
    lines: list[str] = [
        f"# kototok Benchmark Report (vocab_size={vocab_size})",
        "",
        "## 1. Compression (`chars/token`, `bytes/token`, `tokens`)",
        "",
        "| Model | JA c/t | JA b/t | JA tokens | EN c/t | Code c/t | Mixed c/t |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        c = r["compression"]
        lines.append(
            f"| {r['label']} | {c['ja_prose']['chars_per_token']:.3f} | {c['ja_prose']['bytes_per_token']:.3f} | "
            f"{c['ja_prose']['tokens']} | {c['en_prose']['chars_per_token']:.3f} | "
            f"{c['code']['chars_per_token']:.3f} | {c['mixed_ja_code']['chars_per_token']:.3f} |"
        )

    lines += [
        "",
        "## 2. UTF-8 Validity & Dead Token Rate",
        "",
        "| Model | Vocab | Learned Non-Fallback | Invalid UTF-8 | Invalid Rate (%) | Dead Tokens (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        u = r["utf8_audit"]
        lines.append(
            f"| {r['label']} | {r['vocab_size']} | {u['alive_non_fallback_tokens']} | "
            f"{u['invalid_utf8_learned_tokens']} | {u['invalid_utf8_rate_pct']:.2f}% | {r['dead_token_rate_pct']:.2f}% |"
        )

    lines += [
        "",
        "## 3. Japanese Morphological Boundary & Okurigana Evaluation",
        "",
        "| Model | Precision | Recall | F1 | Okurigana Integrity (%) | Noun+Particle Over-Merge (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        jb = r["ja_boundary"]
        lines.append(
            f"| {r['label']} | {jb['boundary_precision']:.4f} | {jb['boundary_recall']:.4f} | "
            f"{jb['boundary_f1']:.4f} | {jb['okurigana_integrity_pct']:.2f}% | {jb['noun_particle_overmerge_pct']:.2f}% |"
        )

    lines += [
        "",
        "## 4. Throughput (1-thread) & Adversarial 1 MiB Latency",
        "",
        "| Model | JA Throughput (MB/s) | Adversarial 1 MiB (ms) | Adversarial Throughput (MB/s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['ja_throughput_mb_s']:.2f} | {r['adversarial_1mb_ms']:.2f} | {r['adversarial_1mb_mb_s']:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def ensure_corpus(data_dir: Path) -> None:
    expected = [
        "train_corpus.txt",
        "test_ja_prose.txt",
        "test_en_prose.txt",
        "test_code.txt",
        "test_mixed_ja_code.txt",
        "adversarial_1mb.txt",
        "ja_boundary_gold.jsonl",
    ]
    if not all((data_dir / f).exists() for f in expected):
        import corpus_builder  # type: ignore

        corpus_builder.DATA_DIR = data_dir
        corpus_builder.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run kototok and baseline evaluations.")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--bin-path", type=Path, default=Path("../target/release/kototok"))
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--train-scale",
        type=float,
        default=1.0,
        help="Multiply train-corpus size (env KOTOTOK_TRAIN_SCALE); test splits stay fixed",
    )
    args = parser.parse_args()

    os.environ["KOTOTOK_TRAIN_SCALE"] = str(args.train_scale)

    out_dir: Path = args.output_dir.resolve()
    data_dir = out_dir / "data"
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Always generate corpus if running --baselines-only or missing
    if args.baselines_only:
        import corpus_builder  # type: ignore

        corpus_builder.DATA_DIR = data_dir
        corpus_builder.main()
    else:
        ensure_corpus(data_dir)

    train_path = data_dir / "train_corpus.txt"
    test_paths = {k: data_dir / v for k, v in TEST_SPLITS.items()}
    test_texts = {k: p.read_text(encoding="utf-8") for k, p in test_paths.items()}
    adv_path = data_dir / "adversarial_1mb.txt"
    adv_text = adv_path.read_text(encoding="utf-8")
    gold_rows = [
        json.loads(line)
        for line in (data_dir / "ja_boundary_gold.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results: list[dict[str, Any]] = []

    if not args.baselines_only:
        bin_path = args.bin_path.resolve()
        if not bin_path.exists():
            raise FileNotFoundError(
                f"kototok binary not found at {bin_path}. Build it or pass --baselines-only."
            )
        for preset in KOTOTOK_PRESETS:
            print(f"[eval] running kototok preset={preset} vocab_size={args.vocab_size}", file=sys.stderr)
            model_path = models_dir / f"kototok_{preset}_{args.vocab_size}.ktok"
            results.append(
                evaluate_kototok_preset(
                    bin_path=bin_path,
                    preset=preset,
                    train_path=train_path,
                    model_path=model_path,
                    vocab_size=args.vocab_size,
                    test_paths=test_paths,
                    test_texts=test_texts,
                    adv_path=adv_path,
                    gold_rows=gold_rows,
                )
            )

    print(f"[eval] training & evaluating hf-bpe vocab_size={args.vocab_size}", file=sys.stderr)
    hf_bpe = train_hf_bpe(train_path, args.vocab_size, models_dir / f"hf_bpe_{args.vocab_size}.json")
    results.append(evaluate_hf_bpe(hf_bpe, test_texts, adv_text, gold_rows))

    print(f"[eval] training & evaluating hf-unigram vocab_size={args.vocab_size}", file=sys.stderr)
    hf_uni = train_hf_unigram(train_path, args.vocab_size, models_dir / f"hf_unigram_{args.vocab_size}.json")
    results.append(evaluate_hf_unigram(hf_uni, test_texts, adv_text, gold_rows))

    print("[eval] evaluating tiktoken cl100k_base (external-100k)", file=sys.stderr)
    results.append(
        evaluate_tiktoken("cl100k_base", "tiktoken:cl100k_base (external-100k)", test_texts, adv_text, gold_rows)
    )

    print("[eval] evaluating tiktoken o200k_base (external-200k)", file=sys.stderr)
    results.append(
        evaluate_tiktoken("o200k_base", "tiktoken:o200k_base (external-200k)", test_texts, adv_text, gold_rows)
    )

    payload = {
        "vocab_size": args.vocab_size,
        "baselines_only": args.baselines_only,
        "results": results,
    }
    results_path = out_dir / "results.json"
    report_path = out_dir / "report.md"
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md = render_markdown_report(results, args.vocab_size)
    report_path.write_text(report_md, encoding="utf-8")

    print(report_md)


if __name__ == "__main__":
    main()
