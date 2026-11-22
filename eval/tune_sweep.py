#!/usr/bin/env python3
"""Hyperparameter sweep for kototok asym-full on the REAL corpus.

Trains variants with different --beta/--picky-threshold/--phase2-min-npmi/
--phase1-ratio, then measures boundary metrics + JA compression + UTF-8
audit. Output: artifacts_tune/sweep_results.json + markdown table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import (  # noqa: E402
    compute_boundary_metrics,
    compression_stats,
    run_cmd,
)

BIN = Path(__file__).resolve().parent.parent / "target" / "release" / "kototok"
DATA = Path(__file__).resolve().parent / "artifacts_real" / "data"
OUT = Path(__file__).resolve().parent / "artifacts_tune"


def evaluate_model(
    model_path: Path,
    gold_rows: list[dict],
    test_ja_text: str,
    test_ja_path: Path,
    test_en_text: str = "",
    test_en_path: Path | None = None,
) -> dict:
    inspect = json.loads(run_cmd([str(BIN), "inspect", "--model", str(model_path), "--json"]))
    enc = json.loads(
        run_cmd([str(BIN), "encode", "--model", str(model_path), "--input", str(test_ja_path), "--json"])
    )
    comp = compression_stats(test_ja_text, enc["token_count"])
    enc_en = {"token_count": 0}
    if test_en_path is not None:
        enc_en = json.loads(
            run_cmd([str(BIN), "encode", "--model", str(model_path), "--input", str(test_en_path), "--json"])
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / "sent.txt"

        def byte_lens_fn(s: str) -> list[int]:
            tmp_input.write_text(s, encoding="utf-8")
            out = json.loads(
                run_cmd([str(BIN), "encode", "--model", str(model_path), "--input", str(tmp_input), "--json"])
            )
            if "byte_lens" in out:
                return [int(x) for x in out["byte_lens"]]
            return [len(p.encode("utf-8", "replace")) for p in out["pieces"]]

        jb = compute_boundary_metrics(gold_rows, byte_lens_fn)

    comp_en = compression_stats(test_en_text, enc_en["token_count"]) if test_en_text else {"chars_per_token": 0.0}
    return {
        "alive_vocab": inspect["alive_vocab_size"],
        "invalid_utf8_pct": round(inspect["invalid_utf8_rate"] * 100.0, 2),
        "ja_chars_per_token": comp["chars_per_token"],
        "en_chars_per_token": comp_en["chars_per_token"],
        "ja_tokens": comp["tokens"],
        "boundary_f1": jb["boundary_f1"],
        "boundary_precision": jb["boundary_precision"],
        "boundary_recall": jb["boundary_recall"],
        "okurigana_pct": jb["okurigana_integrity_pct"],
        "overmerge_pct": jb["noun_particle_overmerge_pct"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--betas", default="0.0,0.5,1.0,2.0,4.0")
    ap.add_argument("--npmis", default="0.25", help="phase2_min_npmi values")
    ap.add_argument("--pickys", default="0.90", help="picky_ios_threshold values")
    ap.add_argument("--ratios", default="0.85", help="phase1_vocab_ratio values")
    ap.add_argument("--begs", default="0.0", help="BEG beg_strength values")
    ap.add_argument("--beg-stem-max", type=float, default=0.7)
    ap.add_argument("--beg-noun-min", type=float, default=1.8)
    ap.add_argument("--beg-particle-lo", type=float, default=6.0)
    ap.add_argument("--beg-particle-hi", type=float, default=9.0)
    ap.add_argument("--beg-phase2s", default="1", help="beg_phase2 0/1 values")
    ap.add_argument("--npmi-asciis", default="", help="phase2_min_npmi_ascii values; empty = preset default")
    args = ap.parse_args()

    betas = [float(x) for x in args.betas.split(",") if x]
    npmis = [float(x) for x in args.npmis.split(",") if x]
    pickys = [float(x) for x in args.pickys.split(",") if x]
    ratios = [float(x) for x in args.ratios.split(",") if x]
    begs = [float(x) for x in args.begs.split(",") if x]
    beg_p2s = [int(x) for x in args.beg_phase2s.split(",") if x]
    npmi_asciis = [float(x) for x in args.npmi_asciis.split(",") if x] or [None]

    train_path = DATA / "train_corpus.txt"
    test_ja_path = DATA / "test_ja_prose.txt"
    test_ja_text = test_ja_path.read_text(encoding="utf-8")
    test_en_path = DATA / "test_en_prose.txt"
    test_en_text = test_en_path.read_text(encoding="utf-8")
    gold_rows = [
        json.loads(l)
        for l in (DATA / "ja_boundary_gold.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    models_dir = OUT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for beta, npmi, picky, ratio, beg, bp2, nasc in itertools.product(
        betas, npmis, pickys, ratios, begs, beg_p2s, npmi_asciis
    ):
        tag = f"b{beta:g}_n{npmi:g}_p{picky:g}_r{ratio:g}" + (
            f"_beg{beg:g}" if beg > 0 else ""
        ) + (f"_bp2-{bp2}" if beg > 0 else "") + (
            f"_na{nasc:g}" if nasc is not None else ""
        )
        model_path = models_dir / f"tune_{tag}_{args.vocab_size}.ktok"
        print(f"[sweep] train {tag}", file=sys.stderr, flush=True)
        cmd = [
            str(BIN), "train",
            "--input", str(train_path),
            "--output", str(model_path),
            "--vocab-size", str(args.vocab_size),
            "--preset", "asym-full",
            "--beta", str(beta),
            "--phase2-min-npmi", str(npmi),
            "--picky-threshold", str(picky),
            "--phase1-ratio", str(ratio),
        ]
        if nasc is not None:
            cmd += ["--phase2-min-npmi-ascii", str(nasc)]
        if beg > 0:
            cmd += [
                "--beg-strength", str(beg),
                "--beg-stem-max", str(args.beg_stem_max),
                "--beg-noun-min", str(args.beg_noun_min),
                "--beg-particle-lo", str(args.beg_particle_lo),
                "--beg-particle-hi", str(args.beg_particle_hi),
                "--beg-phase2", str(bp2),
            ]
        run_cmd(cmd)
        m = evaluate_model(model_path, gold_rows, test_ja_text, test_ja_path,
                           test_en_text, test_en_path)
        m.update({
            "beta": beta, "phase2_min_npmi": npmi, "picky": picky,
            "ratio": ratio, "beg": beg, "beg_phase2": bp2, "npmi_ascii": nasc,
        })
        rows.append(m)
        print(
            f"[sweep] {tag}: F1={m['boundary_f1']:.4f} oku={m['okurigana_pct']:.1f}% "
            f"over={m['overmerge_pct']:.1f}% ja_c/t={m['ja_chars_per_token']:.3f} "
            f"en_c/t={m['en_chars_per_token']:.3f} inv={m['invalid_utf8_pct']}%",
            file=sys.stderr,
            flush=True,
        )

    (OUT / "sweep_results.json").write_text(
        json.dumps({"vocab_size": args.vocab_size, "results": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    hdr = (
        "| beta | npmi | picky | ratio | beg | bp2 | F1 | Oku% | Over% | JA c/t | EN c/t | InvUTF8% |"
    )
    print(hdr)
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in rows:
        print(
            f"| {m['beta']:g} | {m['phase2_min_npmi']:g} | {m['picky']:g} | {m['ratio']:g} | "
            f"{m['beg']:g} | {m['beg_phase2']} | "
            f"{m['boundary_f1']:.4f} | {m['okurigana_pct']:.2f} | {m['overmerge_pct']:.2f} | "
            f"{m['ja_chars_per_token']:.3f} | {m['en_chars_per_token']:.3f} | {m['invalid_utf8_pct']:.2f} |"
        )


if __name__ == "__main__":
    main()
