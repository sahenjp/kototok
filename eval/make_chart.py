#!/usr/bin/env python3
"""Render benchmark results.json into a self-contained SVG chart HTML."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ART = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "artifacts"
RESULTS = ART / "results.json"
OUT = ART / "chart.html"

COLOR_KOTOTOK = "#4f9cf9"
COLOR_BASELINE = "#9aa4b2"
COLOR_EXTERNAL = "#f2a65a"
COLORS = {"kototok": COLOR_KOTOTOK, "baseline": COLOR_BASELINE, "external": COLOR_EXTERNAL}

SHORT = {
    "kototok:asym-full": "asym-full",
    "kototok:asym-no-phase2": "no-phase2",
    "kototok:asym-no-picky": "no-picky",
    "kototok:asym-no-acsg": "no-acsg",
    "kototok:strict-script": "strict-script",
    "kototok:standard-bbpe": "std-bbpe",
    "hf-bpe": "hf-bpe",
    "hf-unigram": "hf-unigram",
    "tiktoken:cl100k_base (external-100k)": "cl100k*",
    "tiktoken:o200k_base (external-200k)": "o200k*",
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_chart(title: str, rows: list[tuple[str, float, str]], fmt: str, higher_better: bool, unit: str) -> str:
    vmax = max(v for _, v, _ in rows) or 1.0
    best = max(v for _, v, _ in rows) if higher_better else min(v for _, v, _ in rows)
    out = [f'<div class="chart"><h3>{esc(title)}</h3>']
    for label, v, kind in rows:
        w = max(0.5, v / vmax * 100.0)
        mark = ' <span class="best">BEST</span>' if v == best else ""
        out.append(
            f'<div class="row"><span class="lbl" title="{esc(label)}">{esc(SHORT.get(label, label))}</span>'
            f'<span class="barwrap"><span class="bar" style="width:{w:.2f}%;background:{COLORS[kind]}"></span></span>'
            f'<span class="val">{v:{fmt}}{esc(unit)}{mark}</span></div>'
        )
    out.append("</div>")
    return "\n".join(out)


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    results = payload["results"]

    def rows_of(key):
        return [(r["label"], key(r), r["kind"]) for r in results]

    charts = [
        bar_chart(
            "JA compression — chars/token (higher = better)",
            rows_of(lambda r: r["compression"]["ja_prose"]["chars_per_token"]),
            ".2f", True, "",
        ),
        bar_chart(
            "JA boundary F1 vs Sudachi gold (higher = better)",
            rows_of(lambda r: r["ja_boundary"]["boundary_f1"]),
            ".3f", True, "",
        ),
        bar_chart(
            "Okurigana integrity % (higher = better)",
            rows_of(lambda r: r["ja_boundary"]["okurigana_integrity_pct"]),
            ".1f", True, "%",
        ),
        bar_chart(
            "Invalid UTF-8 learned tokens % (lower = better)",
            rows_of(lambda r: r["utf8_audit"]["invalid_utf8_rate_pct"]),
            ".2f", False, "%",
        ),
        bar_chart(
            "JA throughput MB/s (higher = better)",
            rows_of(lambda r: r["ja_throughput_mb_s"]),
            ".1f", True, "",
        ),
        bar_chart(
            "Adversarial 1 MiB latency ms (lower = better)",
            rows_of(lambda r: r["adversarial_1mb_ms"]),
            ".1f", False, "",
        ),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>kototok benchmark — vocab_size={payload['vocab_size']}</title>
<style>
body{{font-family:-apple-system,'Segoe UI',sans-serif;background:#0f141a;color:#e6edf3;max-width:960px;margin:24px auto;padding:0 16px}}
h1{{font-size:20px}} h3{{font-size:14px;margin:18px 0 8px;color:#c9d4df}}
.chart{{margin-bottom:8px}}
.row{{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}}
.lbl{{flex:0 0 110px;text-align:right;color:#aeb9c5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.barwrap{{flex:1;background:#1c232c;border-radius:3px;height:16px;display:block}}
.bar{{display:block;height:16px;border-radius:3px}}
.val{{flex:0 0 130px;font-variant-numeric:tabular-nums}}
.best{{color:#7ee787;font-size:10px;font-weight:700}}
.legend{{font-size:12px;color:#aeb9c5;margin:8px 0 20px}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}}
.note{{font-size:11px;color:#6b7683}}
</style></head><body>
<h1>kototok benchmark — vocab_size={payload['vocab_size']} — 10 models</h1>
<div class="legend">
<span class="sw" style="background:{COLOR_KOTOTOK}"></span>kototok
<span class="sw" style="background:{COLOR_BASELINE}"></span>same-vocab baseline (HF)
<span class="sw" style="background:{COLOR_EXTERNAL}"></span>external (tiktoken, 100k/200k vocab — not directly comparable)
</div>
{''.join(charts)}
<p class="note">* cl100k_base / o200k_base are production 100k–200k-vocab tokenizers shown for reference only.
Same-corpus 4k-vocab comparison is among the other 8 rows. Generated from eval/artifacts/results.json.</p>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
