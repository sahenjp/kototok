"""Export a .ktok JSON dump to HuggingFace tokenizer.json.

Usage: kototok dump --model X.ktok --output model.json
       python eval/export_hf.py model.json tokenizer.json

Fidelity note: HF ByteLevel pretokenizer differs from AsymScript pretok;
token counts diverge slightly (JA ~1.4x). Round-trip decode is exact.
"""

import json, sys
d = json.load(open(sys.argv[1]))
toks = {t["internal_id"]: t for t in d["internal_tokens"]}
bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n += 1
b2u = dict(zip(bs, [chr(c) for c in cs]))
def enc(bts): return "".join(b2u[x] for x in bts)
alive = {t["internal_id"] for t in d["internal_tokens"] if t["final_id"] is not None}
pruned = {e["Prune"]["target"] for e in d["events"] if "Prune" in e}
prod = alive - pruned
vocab = {}; next_fid = 0
for t in d["internal_tokens"]:
    if t["final_id"] is not None:
        vocab[enc(t["bytes"])] = t["final_id"]; next_fid = max(next_fid, t["final_id"]+1)
is_result = {e["Merge"]["result"] for e in d["events"] if "Merge" in e}
merges = []
for e in d["events"]:
    if "Merge" in e:
        m = e["Merge"]
        if m["result"] in prod and m["left"] in prod and m["right"] in prod:
            merges.append(f"{enc(toks[m['left']]['bytes'])} {enc(toks[m['right']]['bytes'])}")
# assemble multi-byte atomic tokens that no merge produces
inter = {}; n_atoms = 0
for t in d["internal_tokens"]:
    if t["internal_id"] in prod and t["internal_id"] not in is_result and len(t["bytes"]) > 1:
        bt = t["bytes"]; n_atoms += 1
        for i in range(2, len(bt)+1):
            prefix = enc(bt[:i-1])
            if prefix not in vocab:
                if prefix not in inter:
                    inter[prefix] = next_fid; vocab[prefix] = next_fid; next_fid += 1
            merges.append(f"{prefix} {enc(bt[i-1:i])}")
hf = {
 "version":"1.0","truncation":None,"padding":None,
 "added_tokens":[{"id":fid,"content":tx,"single_word":False,"lstrip":False,"rstrip":False,"normalized":False,"special":True} for tx,fid in d["special_tokens"]],
 "normalizer":None,
 "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":False,"trim_offsets":True,"use_regex":False},
 "post_processor":None,
 "decoder":{"type":"ByteLevel","add_prefix_space":False,"trim_offsets":True,"use_regex":False},
 "model":{"type":"BPE","dropout":None,"unk_token":None,"continuing_subword_prefix":None,"end_of_word_suffix":None,"fuse_unk":False,"byte_fallback":False,"vocab":vocab,"merges":merges}
}
json.dump(hf, open(sys.argv[2],"w"), ensure_ascii=False)
print(f"vocab={len(vocab)} merges={len(merges)} atoms={n_atoms}")
