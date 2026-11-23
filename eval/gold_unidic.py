#!/usr/bin/env python3
"""Generate a UniDic-based boundary gold set for kototok evaluation.

This provides an *independent* segmentation criterion from the Sudachi
SplitMode.C gold (`ja_boundary_gold.jsonl`): sentences are re-segmented with
MeCab/UniDic (`fugashi` + `unidic-lite`), whose lexical units are deliberately
shorter and defined by a different dictionary.

Input is a JSONL file with a ``text`` field per line (e.g. the Sudachi-derived
holdout file), and the output uses the same schema consumed by
``run_eval.compute_boundary_metrics``::

    {"text": str,
     "byte_spans": [[start, end], ...],          # tiles the UTF-8 text
     "okurigana_words": [{"text", "byte_span"}],
     "noun_particles": [{"noun", "particle",
                         "boundary_byte", "combined_byte_span"}]}

The annotation heuristics intentionally mirror the Sudachi gold generator so
that metric semantics stay comparable across the two gold standards:

* ``okurigana_words``: surface of length >= 2 that starts with a CJK kanji,
  ends with hiragana, and is not entirely hiragana.
* ``noun_particles``: a 名詞 token immediately followed by a 助詞 token whose
  surface is at most 2 characters; ``boundary_byte`` is the byte offset of the
  noun/particle seam.

Usage::

    python gold_unidic.py --input /tmp/gold_ja_holdout.jsonl \
        --output /tmp/gold_unidic_holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def is_kanji(c: str) -> bool:
    return "一" <= c <= "鿿" or "㐀" <= c <= "䶿"  # U+4E00-9FFF, U+3400-4DBF


def is_hira(c: str) -> bool:
    return "぀" <= c <= "ゟ"  # U+3040-309F


def is_okuri(w: str) -> bool:
    """Surface looks like a kanji-stem + hiragana-okurigana word."""
    return len(w) >= 2 and is_kanji(w[0]) and is_hira(w[-1]) and not all(is_hira(c) for c in w)


def is_noun(pos1: str | None) -> bool:
    return pos1 == "名詞"


def is_particle(surface: str, pos1: str | None) -> bool:
    return pos1 == "助詞" and len(surface) <= 2


def segment_with_gaps(text: str, tagger: Any) -> list[tuple[str, Any]] | None:
    """Tokenize ``text`` and return (surface, morpheme) entries that tile it.

    MeCab/UniDic drops whitespace instead of emitting it as a token, so raw
    surfaces do not cover the input byte-for-byte.  This aligns each morpheme
    surface to the text in order and re-inserts the skipped inter-token
    substrings as gap entries with ``morpheme=None`` (matching how Sudachi
    emits whitespace as its own gold segment).

    Returns ``None`` if a morpheme surface cannot be aligned (defensive).
    """
    entries: list[tuple[str, Any]] = []
    pos = 0
    for m in tagger(text):
        w = m.surface
        idx = text.find(w, pos)
        if idx < 0:
            return None
        if idx > pos:
            entries.append((text[pos:idx], None))
        entries.append((w, m))
        pos = idx + len(w)
    if pos < len(text):
        entries.append((text[pos:], None))
    return entries


def build_gold_row(text: str, tagger: Any) -> dict[str, Any] | None:
    """Segment ``text`` with UniDic and build one gold row."""
    entries = segment_with_gaps(text, tagger)
    if entries is None or len(entries) < 3:
        return None

    surfaces: list[str] = []
    byte_spans: list[list[int]] = []
    pos_b = 0
    for w, _ in entries:
        b = len(w.encode("utf-8"))
        surfaces.append(w)
        byte_spans.append([pos_b, pos_b + b])
        pos_b += b

    okurigana_words: list[dict[str, Any]] = []
    noun_particles: list[dict[str, Any]] = []
    for i, (w, m) in enumerate(entries):
        if m is None:
            continue
        if is_okuri(w):
            okurigana_words.append({"text": w, "byte_span": byte_spans[i]})
        pos1 = getattr(m.feature, "pos1", None)
        # Adjacent entries are both real morphemes only when nothing was
        # dropped between them, so index adjacency implies text adjacency.
        if i + 1 < len(entries) and is_noun(pos1):
            next_m = entries[i + 1][1]
            if next_m is not None and is_particle(
                surfaces[i + 1], getattr(next_m.feature, "pos1", None)
            ):
                noun_particles.append(
                    {
                        "noun": w,
                        "particle": surfaces[i + 1],
                        "boundary_byte": byte_spans[i][1],
                        "combined_byte_span": [byte_spans[i][0], byte_spans[i + 1][1]],
                    }
                )

    return {
        "text": text,
        "byte_spans": byte_spans,
        "okurigana_words": okurigana_words,
        "noun_particles": noun_particles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate a boundary gold JSONL with UniDic (fugashi) segmentation.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source JSONL; each line must contain a 'text' field.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination gold JSONL in the run_eval schema.",
    )
    args = parser.parse_args()

    try:
        from fugashi import Tagger
    except ImportError:
        sys.exit(
            "fugashi is required (e.g. `uv pip install fugashi unidic-lite`)."
        )

    tagger = Tagger()

    rows_out: list[dict[str, Any]] = []
    n_in = 0
    n_skipped = 0
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            text = json.loads(line)["text"]
            row = build_gold_row(text, tagger)
            if row is None:
                n_skipped += 1
                continue
            rows_out.append(row)

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_oku = sum(len(r["okurigana_words"]) for r in rows_out)
    n_np = sum(len(r["noun_particles"]) for r in rows_out)
    print(
        f"{len(rows_out)}/{n_in} rows -> {args.output} "
        f"(skipped={n_skipped}, okurigana_words={n_oku}, noun_particles={n_np})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
