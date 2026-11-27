"""Export a kototok model to Hugging Face tokenizer.json.

Prefer the native Rust exporter:

    kototok export --model X.ktok --output tokenizer.json

The native path fixes PickyBPE intermediate pruning and adds AsymScript
hard-boundary pretokenization. This script is a compatibility wrapper: when a
`kototok` binary is available it delegates to `kototok export --format hf`;
otherwise it falls back to the legacy kototok-JSON converter.

Usage: python eval/export_hf.py <input> <output>

`<input>` may be a `.ktok` model or a kototok `.json` dump. The legacy
fallback also accepts both, parsing the small `.ktok` container directly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_kototok_binary() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("target/debug/kototok", "target/release/kototok"):
        path = root / rel
        if path.is_file() and os.access(path, os.X_OK):
            return path
    found = shutil.which("kototok")
    return Path(found) if found else None


def byte_level_encoder():
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
    b2u = dict(zip(bs, [chr(c) for c in cs]))

    def enc(bts):
        return "".join(b2u[x] for x in bts)

    return enc


def load_kototok_model_json(input_path: Path):
    data = input_path.read_bytes()
    if data.startswith(b"KTOK"):
        if len(data) < 16:
            raise ValueError(f"{input_path} is too short for a .ktok header")
        version = int.from_bytes(data[4:8], "little")
        if version != 1:
            raise ValueError(f"unsupported .ktok format version {version}")
        payload_len = int.from_bytes(data[8:16], "little")
        payload = data[16:]
        if len(payload) != payload_len:
            raise ValueError(f"{input_path} has a truncated or trailing .ktok payload")
        return json.loads(payload)
    return json.loads(data.decode("utf-8"))


def export_legacy_json(input_path: Path, output_path: Path) -> None:
    d = load_kototok_model_json(input_path)
    if "internal_tokens" not in d or "events" not in d:
        raise ValueError(
            f"{input_path} is not a kototok model/dump; install/build kototok "
            "and use the native exporter for HF inputs"
        )

    toks = {t["internal_id"]: t for t in d["internal_tokens"]}
    enc = byte_level_encoder()
    alive = {t["internal_id"] for t in d["internal_tokens"] if t["final_id"] is not None}
    pruned = {e["Prune"]["target"] for e in d["events"] if "Prune" in e}
    prod = alive - pruned

    vocab = {}
    next_fid = 0
    for t in d["internal_tokens"]:
        if t["final_id"] is not None:
            vocab[enc(t["bytes"])] = t["final_id"]
            next_fid = max(next_fid, t["final_id"] + 1)

    is_result = {e["Merge"]["result"] for e in d["events"] if "Merge" in e}
    merges = []
    for e in d["events"]:
        if "Merge" in e:
            m = e["Merge"]
            if m["result"] in prod and m["left"] in prod and m["right"] in prod:
                merges.append(
                    f"{enc(toks[m['left']]['bytes'])} {enc(toks[m['right']]['bytes'])}"
                )

    # assemble multi-byte atomic tokens that no merge produces
    inter = {}
    n_atoms = 0
    for t in d["internal_tokens"]:
        if t["internal_id"] in prod and t["internal_id"] not in is_result and len(t["bytes"]) > 1:
            bt = t["bytes"]
            n_atoms += 1
            for i in range(2, len(bt) + 1):
                prefix = enc(bt[: i - 1])
                if prefix not in vocab:
                    if prefix not in inter:
                        inter[prefix] = next_fid
                        vocab[prefix] = next_fid
                        next_fid += 1
                merges.append(f"{prefix} {enc(bt[i - 1:i])}")

    hf = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": fid,
                "content": tx,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for tx, fid in d["special_tokens"]
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(hf, f, ensure_ascii=False)
    print(f"vocab={len(vocab)} merges={len(merges)} atoms={n_atoms}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python eval/export_hf.py <input> <output>", file=sys.stderr)
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])

    binary = find_kototok_binary()
    if binary is not None:
        try:
            subprocess.run(
                [
                    str(binary),
                    "export",
                    "--model",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--format",
                    "hf",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            return e.returncode
        except OSError as e:
            print(f"error: failed to invoke {binary}: {e}", file=sys.stderr)
            return 1
        return 0

    try:
        export_legacy_json(input_path, output_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
