#!/usr/bin/env python3
"""Minimal tokenizer comparison server for kototok vs baselines.

Stdlib only (http.server). Serves index.html and
GET /api/tokenize?text=... -> JSON {models: [{name, count, cps, pieces}]}.

Run:  ./.venv/bin/python web/server.py   (from eval/)
Then open http://127.0.0.1:8777
"""

import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = EVAL_DIR / "artifacts_real" / "models"
WEB_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(EVAL_DIR.parent / "pylib"))

PORT = 8777

loaders = []


def load_models():
    import kototok
    from tokenizers import Tokenizer as HFTokenizer
    import tiktoken
    import sentencepiece as spm

    def ktok(name, fname):
        tok = kototok.Tokenizer.load(str(MODELS_DIR / fname))
        return (name, lambda text: tok.pieces(text))

    def hf(name, fname):
        p = fname if str(fname).startswith("/") else MODELS_DIR / fname
        tok = HFTokenizer.from_file(str(p))
        return (name,
                lambda text: [tok.decode([i]) for i in tok.encode(text).ids])

    def tik(name, enc_name):
        enc = tiktoken.get_encoding(enc_name)

        def run(text):
            out = []
            for tid in enc.encode(text):
                b = enc.decode_single_token_bytes(tid)
                try:
                    out.append(b.decode("utf-8"))
                except UnicodeDecodeError:
                    out.append(" ".join(f"{x:02x}" for x in b))
            return out

        return (name, run)

    def sp(name, fname):
        sp_tok = spm.SentencePieceProcessor(model_file=str(fname))
        return (name,
                lambda text: [p.replace("▁", "␣") for p in sp_tok.encode(text, out_type=str)])

    loaders.append(ktok("kototok 200k (s3o-BKO)", "kototok_s3o_g_200k.ktok"))
    EXT = EVAL_DIR / "artifacts_real" / "external"
    loaders.append(sp("sarashina2 (102k)", EXT / "sarashina2_tokenizer.model"))
    loaders.append(hf("llm-jp", EXT / "llmjp_tokenizer.json"))
    loaders.append(sp("plamo (32k)", EXT / "plamo_tokenizer.model"))
    loaders.append(hf("qwen2.5 (151k)", EXT / "qwen25_tokenizer.json"))
    loaders.append(hf("llama3 (128k)", EXT / "llama3_tokenizer.json"))
    loaders.append(hf("deepseek-v3", EXT / "deepseek_tokenizer.json"))
    loaders.append(sp("gemma2 (256k)", EXT / "gemma2_tokenizer.model"))
    loaders.append(sp("mistral (32k)", EXT / "mistral_tokenizer.model"))
    loaders.append(tik("tiktoken cl100k", "cl100k_base"))
    loaders.append(tik("tiktoken o200k", "o200k_base"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8",
                       (WEB_DIR / "index.html").read_bytes())
            return
        if url.path == "/api/tokenize":
            qs = urllib.parse.parse_qs(url.query)
            texts = {
                "ja": qs.get("ja", [""])[0],
                "en": qs.get("en", [""])[0],
                "mix": qs.get("mix", [""])[0],
            }
            models = []
            for name, fn in loaders:
                m = {"name": name}
                for lang, text in texts.items():
                    try:
                        pieces = fn(text)
                        n = max(len(pieces), 1)
                        m[lang] = {
                            "count": len(pieces),
                            "cps": round(len(text) / n, 3),
                            "pieces": pieces,
                        }
                    except Exception as e:  # noqa: BLE001
                        m[lang] = {"error": str(e)}
                models.append(m)
            self._send(200, "application/json; charset=utf-8",
                       json.dumps({"models": models}, ensure_ascii=False))
            return
        self._send(404, "text/plain", "not found")


def main():
    load_models()
    print(f"serving on http://127.0.0.1:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
