#!/usr/bin/env python3
"""Extract clean Japanese prose from a jawiki pages-articles bz2 dump.

Streams the XML, keeps ns=0 pages, strips wikitext markup, and writes
paragraph-joined plain text. Cheap regex-based cleaning is fine for a
tokenizer training corpus (we only need natural-frequency Japanese).

Usage: python3 jawiki_extract.py /tmp/jawiki_p1.bz2 /tmp/jawiki_p1.txt
"""
import bz2
import re
import sys

RE_TAG = re.compile(r"<[^>]+>")
RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
RE_HEADING = re.compile(r"^=+(.+?)=+\s*$", re.M)
RE_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
RE_FILE = re.compile(r"\[\[(?:ファイル|File|Image|画像|Category|カテゴリ)[^\]]*\]\]")
RE_EXTLINK = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
RE_URL = re.compile(r"https?://\S+")
RE_LIST = re.compile(r"^[*#:;]+\s*", re.M)
RE_QUOTE = re.compile(r"'{2,}")
RE_JA = re.compile(r"[぀-ヿ一-鿿]")


def clean(text: str) -> str:
    text = RE_COMMENT.sub(" ", text)
    text = RE_TABLE.sub(" ", text)
    for _ in range(4):
        text = RE_TEMPLATE.sub(" ", text)
    text = RE_FILE.sub(" ", text)
    text = RE_TAG.sub(" ", text)
    text = RE_EXTLINK.sub(r"\1", text)
    text = RE_URL.sub(" ", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_HEADING.sub(r"\n\1\n", text)
    text = RE_LIST.sub("", text)
    text = RE_QUOTE.sub("", text)
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if len(ln) < 20:
            continue
        ja = len(RE_JA.findall(ln))
        if ja < max(8, len(ln) // 3):
            continue
        lines.append(ln)
    return "\n".join(lines)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    out = open(dst, "w", encoding="utf-8")
    buf: list[str] = []
    in_page = in_text = False
    keep = False
    title = ""
    n_docs = n_bytes = 0
    with bz2.open(src, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s == "<page>":
                in_page = True
                buf = []
                keep = False
                title = ""
            elif s == "</page>":
                if keep and buf:
                    body = clean("".join(buf))
                    if len(body) >= 400:
                        out.write(body + "\n\n")
                        n_docs += 1
                        n_bytes += len(body.encode())
                in_page = in_text = False
            elif in_page and s.startswith("<title>"):
                title = s[7:-8] if s.endswith("</title>") else s[7:]
            elif in_page and s.startswith("<ns>"):
                keep = s == "<ns>0</ns>"
            elif in_page and s.startswith("<text"):
                in_text = True
                rest = s.split(">", 1)[1] if ">" in s else ""
                if rest.endswith("</text>"):
                    buf.append(rest[:-7])
                    in_text = False
                elif rest:
                    buf.append(rest)
            elif in_text:
                if s.endswith("</text>"):
                    buf.append(line[: line.rindex("</text>")])
                    in_text = False
                else:
                    buf.append(line)
            if n_docs % 5000 == 0 and n_docs and not getattr(main, "_p", False):
                pass
            if n_docs and n_docs % 10000 == 0 and main._m != n_docs:
                main._m = n_docs
                print(f"[extract] {n_docs} docs {n_bytes / 1e6:.0f}MB",
                      file=sys.stderr, flush=True)
    out.close()
    print(f"[done] {dst} {n_docs} docs {n_bytes / 1e6:.1f}MB", file=sys.stderr)


main._m = 0
main()
