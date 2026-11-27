#!/usr/bin/env python3
"""68-Domain Global Multi-Domain Tokenizer Championship Benchmark.

Evaluates tokenizers across all 68 domains (Language: English-Edu + 57 global languages,
Mathematics, and 9 Code languages) and compares total token counts (lower is better).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import tiktoken
from tokenizers import Tokenizer as HfTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pylib"))
import kototok  # noqa: E402

LANGS_57 = [
    "jpn_Jpan", "cmn_Hani", "kor_Hang", "rus_Cyrl", "deu_Latn", "spa_Latn",
    "fra_Latn", "ita_Latn", "por_Latn", "pol_Latn", "nld_Latn", "ind_Latn",
    "vie_Latn", "fas_Arab", "arb_Arab", "tur_Latn", "tha_Thai", "ukr_Cyrl",
    "ell_Grek", "ces_Latn", "swe_Latn", "hun_Latn", "ron_Latn", "nob_Latn",
    "dan_Latn", "bul_Cyrl", "fin_Latn", "hin_Deva", "ben_Beng", "slk_Latn",
    "heb_Hebr", "lit_Latn", "bos_Latn", "slv_Latn", "ekk_Latn", "cat_Latn",
    "tam_Taml", "hrv_Latn", "lvs_Latn", "zsm_Latn", "azj_Latn", "srp_Cyrl",
    "kat_Geor", "npi_Deva", "mar_Deva", "mal_Mlym", "kaz_Cyrl", "urd_Arab",
    "als_Latn", "mkd_Cyrl", "tel_Telu", "kan_Knda", "mya_Mymr", "guj_Gujr",
    "khm_Khmr", "lao_Laoo", "swh_Latn",
]

CODE_9 = [
    "Code-Python", "Code-C", "Code-C++", "Code-Rust", "Code-Java",
    "Code-JavaScript", "Code-Go", "Code-HTML", "Code-SQL",
]

ALL_68 = ["English-Edu"] + LANGS_57 + ["Mathematics"] + CODE_9


def load_models(specs: list[str]) -> list[tuple[str, Any, int]]:
    loaded = []
    for spec in specs:
        label, kind_path = spec.split("=", 1)
        if kind_path.startswith("tiktoken:"):
            enc_name = kind_path.split(":", 1)[1]
            enc = tiktoken.get_encoding(enc_name)
            loaded.append((label, lambda t, e=enc: len(e.encode(t, disallowed_special=())), enc.n_vocab))
        elif kind_path.startswith("hf:"):
            p = kind_path.split(":", 1)[1]
            ht = HfTokenizer.from_file(p)
            loaded.append((label, lambda t, h=ht: len(h.encode(t, add_special_tokens=False).ids), ht.get_vocab_size()))
        elif kind_path.startswith("ktok:"):
            rest = kind_path.split(":", 1)[1]
            if "@" in rest:
                p, mode = rest.rsplit("@", 1)
                if mode == "compress":
                    tok = kototok.Tokenizer.load(p, tags=[])
                elif mode == "balanced":
                    tok = kototok.Tokenizer.load(p, tags=["ja", "casual", "kozh", "okuri"])
                else:
                    tok = kototok.Tokenizer.load(p, tags=[x.strip() for x in mode.split(",") if x.strip()])
            else:
                tok = kototok.Tokenizer.load(rest)
            loaded.append((label, lambda t, k=tok: len(k.encode(t)), tok.vocab_size))
        else:
            tok = kototok.Tokenizer.load(kind_path)
            loaded.append((label, lambda t, k=tok: len(k.encode(t)), tok.vocab_size))
    return loaded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("eval/global_68/eval"))
    ap.add_argument("--out-json", type=Path, default=Path("eval/global_68/bench_results.json"))
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: Label=ktok:path[@mode] | Label=hf:path | Label=tiktoken:o200k_base",
    )
    args = ap.parse_args()

    models = load_models(args.model)
    domains: dict[str, str] = {}
    for d in ALL_68:
        p = args.eval_dir / f"{d}.txt"
        if not p.is_file():
            raise FileNotFoundError(f"Missing eval domain file: {p}")
        domains[d] = p.read_text(encoding="utf-8")

    results: dict[str, dict[str, Any]] = {}
    for label, fn, vs in models:
        per_dom: dict[str, int] = {}
        for d in ALL_68:
            per_dom[d] = fn(domains[d])
        lang_sum = per_dom["English-Edu"] + sum(per_dom[l] for l in LANGS_57)
        math_sum = per_dom["Mathematics"]
        code_sum = sum(per_dom[c] for c in CODE_9)
        results[label] = {
            "vocab_size": vs,
            "language_total": lang_sum,
            "math_total": math_sum,
            "code_total": code_sum,
            "grand_total": lang_sum + math_sum + code_sum,
            "domains": per_dom,
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 68-DOMAIN CHAMPIONSHIP SUMMARY (Lower = Better) ===")
    print(f"{'Model':<28} {'Vocab':>8} {'Language':>11} {'Math':>9} {'Code':>9} {'GrandTotal':>11}")
    print("-" * 82)
    for label, r in sorted(results.items(), key=lambda kv: kv[1]["grand_total"]):
        print(
            f"{label:<28} {r['vocab_size']:>8,} {r['language_total']:>11,} "
            f"{r['math_total']:>9,} {r['code_total']:>9,} {r['grand_total']:>11,}"
        )

    # Show win counts across all 68 domains
    wins = {label: 0 for label in results}
    for d in ALL_68:
        best = min(results.keys(), key=lambda lb: results[lb]["domains"][d])
        wins[best] += 1
    print("\nDomain wins (out of 68):", wins)
    print(f"\nKey rows (jpn_Jpan / cmn_Hani / kor_Hang / English-Edu / Mathematics / Code-Python):")
    for label, r in results.items():
        dm = r["domains"]
        print(
            f"  {label:<26}: JA={dm['jpn_Jpan']:,} ZH={dm['cmn_Hani']:,} KO={dm['kor_Hang']:,} "
            f"EN={dm['English-Edu']:,} Math={dm['Mathematics']:,} Py={dm['Code-Python']:,}"
        )


if __name__ == "__main__":
    main()
