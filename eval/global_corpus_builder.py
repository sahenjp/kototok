#!/usr/bin/env python3
"""Build a 68-domain multilingual + math + code training corpus and strictly disjoint held-out benchmark.

Sources (all public, ungated HuggingFace datasets):
- English-Edu: HuggingFaceFW/fineweb-edu (sample/10BT shard 0 train, shard 1 eval)
- 57 languages (jpn_Jpan .. swh_Latn): HuggingFaceFW/fineweb-2 (train/000_00000 vs test/000_00000)
- Mathematics: HuggingFaceTB/finemath (finemath-4plus shard 0 train, shard 1 eval)
- 9 Code languages: bigcode/the-stack-smol-xl (disjoint prefix eval vs subsequent docs train)

Uses HTTP Range reads on Parquet files so only the required row groups are transferred.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests

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
    ("Code-Python", "python", 320_000, 7_000_000),
    ("Code-C", "c", 160_000, 4_500_000),
    ("Code-C++", "c++", 140_000, 4_500_000),
    ("Code-Rust", "rust", 140_000, 4_500_000),
    ("Code-Java", "java", 260_000, 5_500_000),
    ("Code-JavaScript", "javascript", 200_000, 5_000_000),
    ("Code-Go", "go", 180_000, 4_500_000),
    ("Code-HTML", "html", 160_000, 4_500_000),
    ("Code-SQL", "sql", 180_000, 4_500_000),
]

EVAL_BYTES_LANG = {
    "jpn_Jpan": 700_000,
    "cmn_Hani": 700_000,
    "kor_Hang": 450_000,
}
DEFAULT_EVAL_BYTES_LANG = 90_000

TRAIN_BYTES_LANG = {
    "jpn_Jpan": 20_000_000,  # supplemented with 14MB local jawiki -> 34MB total JA
    "cmn_Hani": 26_000_000,
    "kor_Hang": 20_000_000,
}
DEFAULT_TRAIN_BYTES_LANG = 2_200_000

# Tier-1 balanced global mixture (strictly disjoint FineWeb-2 train shards; eval/ untouched).
BALANCED_TRAIN_BYTES_LANG = {
    "jpn_Jpan": 106_000_000,  # + 8MB jawiki -> 114MB total JA
    "cmn_Hani": 96_000_000,
    "kor_Hang": 52_000_000,
    "ben_Beng": 38_000_000,
    "tha_Thai": 22_000_000,
    "hin_Deva": 21_000_000,
    "rus_Cyrl": 20_000_000,
    "deu_Latn": 18_000_000,
    "arb_Arab": 17_000_000,
    "tam_Taml": 14_000_000,
    "mar_Deva": 12_000_000,
    "por_Latn": 12_500_000,
    "spa_Latn": 12_500_000,
    "nld_Latn": 11_500_000,
    "fra_Latn": 11_500_000,
    "kat_Geor": 9_200_000,
    "ita_Latn": 8_000_000,
    "ell_Grek": 7_500_000,
    "tur_Latn": 6_800_000,
    "ces_Latn": 6_800_000,
    "pol_Latn": 6_800_000,
    "ron_Latn": 6_800_000,
    "lit_Latn": 6_800_000,
    "fin_Latn": 6_800_000,
    "heb_Hebr": 6_500_000,
    "kaz_Cyrl": 6_200_000,
    "hun_Latn": 5_800_000,
    "ekk_Latn": 5_800_000,
    "slk_Latn": 5_800_000,
    "ind_Latn": 5_800_000,
    "azj_Latn": 5_800_000,
    "bos_Latn": 5_800_000,
    "ukr_Cyrl": 5_800_000,
    "hrv_Latn": 5_800_000,
    "swh_Latn": 5_800_000,
    "lvs_Latn": 5_800_000,
    "fas_Arab": 5_200_000,
    "zsm_Latn": 5_200_000,
    "swe_Latn": 5_200_000,
    "mkd_Cyrl": 5_200_000,
    "slv_Latn": 5_200_000,
    "als_Latn": 5_200_000,
    "npi_Deva": 5_200_000,
    "bul_Cyrl": 5_200_000,
    "tel_Telu": 5_200_000,
    "srp_Cyrl": 5_000_000,
    "mal_Mlym": 5_200_000,
    "guj_Gujr": 5_200_000,
    "kan_Knda": 5_000_000,
    "vie_Latn": 5_000_000,
    "dan_Latn": 4_800_000,
    "nob_Latn": 4_800_000,
    "urd_Arab": 4_800_000,
    "cat_Latn": 4_500_000,
}

ENGLISH_EVAL_BYTES = 1_100_000
ENGLISH_TRAIN_BYTES = 24_000_000
MATH_EVAL_BYTES = 850_000
MATH_TRAIN_BYTES = 18_000_000


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range requests for Parquet reading."""

    def __init__(self, url: str, timeout: int = 45) -> None:
        self.url = url
        self.timeout = timeout
        self.sess = requests.Session()
        h = self.sess.head(url, allow_redirects=True, timeout=timeout)
        h.raise_for_status()
        self.url = h.url
        self.size = int(h.headers["Content-Length"])
        self.pos = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: Any) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(4):
            try:
                r = self.sess.get(
                    self.url,
                    headers={"Range": f"bytes={self.pos}-{end}"},
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.content
                n = len(data)
                b[:n] = data
                self.pos += n
                self.bytes_fetched += n
                return n
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return 0


def read_parquet_texts(
    url: str,
    target_bytes: int,
    column: str = "text",
    skip_docs: int = 0,
    max_doc_bytes: int = 120_000,
) -> tuple[list[str], int]:
    """Stream text documents from a remote Parquet file up to `target_bytes`."""
    raw = HttpRangeFile(url)
    pf = pq.ParquetFile(io.BufferedReader(raw, buffer_size=256 * 1024))
    docs: list[str] = []
    total = 0
    seen = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[column])
        for val in tbl[column].to_pylist():
            if not isinstance(val, str) or not val.strip():
                continue
            seen += 1
            if seen <= skip_docs:
                continue
            b = val.encode("utf-8", errors="ignore")
            if len(b) > max_doc_bytes:
                val = b[:max_doc_bytes].decode("utf-8", errors="ignore")
                b = val.encode("utf-8")
            docs.append(val)
            total += len(b) + 1
            if total >= target_bytes:
                return docs, seen
    return docs, seen


def write_docs(path: Path, docs: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d)
            if not d.endswith("\n"):
                f.write("\n")
    tmp.replace(path)
    return path.stat().st_size


def fetch_fineweb2_lang(
    lang: str,
    eval_dir: Path,
    train_dir: Path,
    train_bytes_map: dict[str, int] | None = None,
) -> str:
    ev_path = eval_dir / f"{lang}.txt"
    tr_path = train_dir / f"{lang}.txt"
    ev_target = EVAL_BYTES_LANG.get(lang, DEFAULT_EVAL_BYTES_LANG)
    tb_map = train_bytes_map if train_bytes_map is not None else TRAIN_BYTES_LANG
    tr_target = tb_map.get(lang, DEFAULT_TRAIN_BYTES_LANG)

    if ev_path.is_file() and ev_path.stat().st_size >= ev_target * 0.5 and tr_path.is_file() and tr_path.stat().st_size >= tr_target * 0.85:
        return f"cached {lang} (ev={ev_path.stat().st_size//1024}KB tr={tr_path.stat().st_size//1024}KB)"

    base = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/{lang}"
    train_url = f"{base}/train/000_00000.parquet"
    test_url = f"{base}/test/000_00000.parquet"

    skip_train = 0
    if ev_path.is_file() and ev_path.stat().st_size >= ev_target * 0.5:
        ev_sz = ev_path.stat().st_size
    else:
        try:
            ev_docs, _ = read_parquet_texts(test_url, ev_target)
        except Exception:
            # Rare fallback if test shard unavailable: take disjoint prefix of train
            ev_docs, skip_train = read_parquet_texts(train_url, ev_target)
        ev_sz = write_docs(ev_path, ev_docs)

    tr_docs, _ = read_parquet_texts(train_url, tr_target, skip_docs=skip_train)
    # If shard 0 was smaller than tr_target, pull subsequent shards
    shard_idx = 1
    while sum(len(d.encode()) for d in tr_docs) < tr_target * 0.85 and shard_idx <= 4:
        try:
            rem = tr_target - sum(len(d.encode()) for d in tr_docs)
            more, _ = read_parquet_texts(f"{base}/train/000_{shard_idx:05d}.parquet", rem)
            if not more:
                break
            tr_docs.extend(more)
            shard_idx += 1
        except Exception:
            break
    tr_sz = write_docs(tr_path, tr_docs)
    return f"ok {lang} (ev={ev_sz//1024}KB tr={tr_sz//1024}KB)"


def fetch_english_edu(eval_dir: Path, train_dir: Path) -> str:
    ev_path = eval_dir / "English-Edu.txt"
    tr_path = train_dir / "English-Edu.txt"
    if ev_path.is_file() and tr_path.is_file() and tr_path.stat().st_size >= ENGLISH_TRAIN_BYTES * 0.8:
        return f"cached English-Edu (ev={ev_path.stat().st_size//1024}KB tr={tr_path.stat().st_size//1024}KB)"
    base = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
    ev_docs, _ = read_parquet_texts(f"{base}/001_00000.parquet", ENGLISH_EVAL_BYTES)
    tr_docs, _ = read_parquet_texts(f"{base}/000_00000.parquet", ENGLISH_TRAIN_BYTES)
    return f"ok English-Edu (ev={write_docs(ev_path, ev_docs)//1024}KB tr={write_docs(tr_path, tr_docs)//1024}KB)"


def fetch_math(eval_dir: Path, train_dir: Path) -> str:
    ev_path = eval_dir / "Mathematics.txt"
    tr_path = train_dir / "Mathematics.txt"
    if ev_path.is_file() and tr_path.is_file() and tr_path.stat().st_size >= MATH_TRAIN_BYTES * 0.8:
        return f"cached Mathematics (ev={ev_path.stat().st_size//1024}KB tr={tr_path.stat().st_size//1024}KB)"
    base = "https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/main/finemath-4plus"
    ev_docs, _ = read_parquet_texts(f"{base}/train-00001-of-00064.parquet", MATH_EVAL_BYTES)
    tr_docs, _ = read_parquet_texts(f"{base}/train-00000-of-00064.parquet", MATH_TRAIN_BYTES)
    return f"ok Mathematics (ev={write_docs(ev_path, ev_docs)//1024}KB tr={write_docs(tr_path, tr_docs)//1024}KB)"


def fetch_code_lang(domain: str, folder: str, ev_target: int, tr_target: int, eval_dir: Path, train_dir: Path) -> str:
    ev_path = eval_dir / f"{domain}.txt"
    tr_path = train_dir / f"{domain}.txt"
    if ev_path.is_file() and tr_path.is_file() and tr_path.stat().st_size >= tr_target * 0.8:
        return f"cached {domain} (ev={ev_path.stat().st_size//1024}KB tr={tr_path.stat().st_size//1024}KB)"

    url = f"https://huggingface.co/datasets/bigcode/the-stack-smol-xl/resolve/main/data/{folder}/data.json"
    ev_docs: list[str] = []
    tr_docs: list[str] = []
    ev_bytes = 0
    tr_bytes = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            text = obj.get("content") or obj.get("code") or obj.get("text") or ""
            if not text.strip():
                continue
            b = text.encode("utf-8", errors="ignore")
            if len(b) > 80_000:
                text = b[:80_000].decode("utf-8", errors="ignore")
                b = text.encode("utf-8")
            if ev_bytes < ev_target:
                ev_docs.append(text)
                ev_bytes += len(b) + 1
            elif tr_bytes < tr_target:
                tr_docs.append(text)
                tr_bytes += len(b) + 1
            else:
                break
    return f"ok {domain} (ev={write_docs(ev_path, ev_docs)//1024}KB tr={write_docs(tr_path, tr_docs)//1024}KB)"


def assemble_train_corpus(
    train_dir: Path,
    out_file: Path,
    jawiki_extra: Path | None,
    jawiki_bytes: int = 14_000_000,
) -> int:
    """Concatenate all 68 domain train slices (+ optional jawiki slice) into one training file."""
    tmp = out_file.with_suffix(".txt.tmp")
    total = 0
    with open(tmp, "wb") as out:
        for p in sorted(train_dir.glob("*.txt")):
            data = p.read_bytes()
            out.write(data)
            if data and not data.endswith(b"\n"):
                out.write(b"\n")
            total += len(data)
        if jawiki_extra and jawiki_extra.is_file() and jawiki_bytes > 0:
            # Add local jawiki for encyclopedic JA richness alongside FineWeb-2 web JA
            with open(jawiki_extra, "rb") as jf:
                chunk = jf.read(jawiki_bytes)
                # Trim to last newline to keep UTF-8 valid
                nl = chunk.rfind(b"\n")
                if nl > 0:
                    chunk = chunk[: nl + 1]
                out.write(chunk)
                total += len(chunk)
    tmp.replace(out_file)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("eval/global_68"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jawiki", type=Path, default=Path("eval/artifacts_real/data/train_corpus_jw.txt"))
    ap.add_argument("--balanced", action="store_true", help="Build Tier-1 balanced 68-domain corpus")
    args = ap.parse_args()

    eval_dir = args.out_dir / "eval"
    train_dir = args.out_dir / ("train_slices_balanced" if args.balanced else "train_slices")
    eval_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # In balanced mode, reuse existing unchanged slices from train_slices/ via hardlink/copy
    if args.balanced:
        base_train = args.out_dir / "train_slices"
        if base_train.is_dir():
            for src in base_train.glob("*.txt"):
                dst = train_dir / src.name
                if not dst.exists():
                    lang_key = src.stem
                    need = BALANCED_TRAIN_BYTES_LANG.get(lang_key, DEFAULT_TRAIN_BYTES_LANG)
                    if lang_key not in BALANCED_TRAIN_BYTES_LANG or src.stat().st_size >= need * 0.85:
                        os.link(src, dst)

    train_map = BALANCED_TRAIN_BYTES_LANG if args.balanced else TRAIN_BYTES_LANG
    t0 = time.time()
    tasks: list[tuple[str, Any, tuple[Any, ...]]] = [
        ("English-Edu", fetch_english_edu, (eval_dir, train_dir)),
        ("Mathematics", fetch_math, (eval_dir, train_dir)),
    ]
    for domain, folder, ev_b, tr_b in CODE_9:
        tasks.append((domain, fetch_code_lang, (domain, folder, ev_b, tr_b, eval_dir, train_dir)))
    for lang in LANGS_57:
        tasks.append((lang, fetch_fineweb2_lang, (lang, eval_dir, train_dir, train_map)))

    ok = 0
    failed: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fn, *fargs): name for name, fn, fargs in tasks}
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            try:
                msg = fut.result()
                ok += 1
                print(f"[{ok:2d}/{len(tasks)}] {msg}", flush=True)
            except Exception as e:
                failed.append((name, str(e)))
                print(f"[ERR] {name}: {e}", file=sys.stderr, flush=True)

    if failed:
        print(f"FAILED {len(failed)} domains: {failed}", file=sys.stderr)
        sys.exit(1)

    train_out = args.out_dir / ("train_global_68_balanced.txt" if args.balanced else "train_global_68.txt")
    jawiki_bytes = 8_000_000 if args.balanced else 14_000_000
    total_train = assemble_train_corpus(train_dir, train_out, args.jawiki, jawiki_bytes=jawiki_bytes)
    total_eval = sum(p.stat().st_size for p in eval_dir.glob("*.txt"))
    print(
        f"DONE in {time.time()-t0:.1f}s: 68 domains built | "
        f"eval={total_eval/1e6:.2f}MB | train={total_train/1e6:.2f}MB -> {train_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
