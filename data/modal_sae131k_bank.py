"""Modal app `maemm-27b-sae131k-bank`: IN-DISTRIBUTION SAE banks for Qwen3.6-27B from our own layer-42 131k SAE (/data/sae/ae.pt,
ceselder/qwen36-27b-sae-l42: 131,072 features, k=64) and its fresh max-activating examples (/data/sae/maxacts_fresh.pt: [131072, 16, 32]
tokens + activations from a 41M-token scan of the fresh acts store, 32-token contexts). The 131k SAE is the eval SAE of the v3 eval
cache ("sae" family), so training on its features (eval features excluded) makes that eval in-distribution -- the 27B analogue of the
Qwen3-8B chain (trained + evaluated on Karvonen's 65k SAE). Same construction as data/modal_sae8b_bank.py:

    split()  eval = EVERY 131k feature in /data/pool_heldout (family "sae": 13,107 features; the v3 cache's 512 are a subset), then of
             the remaining live features (top max-act > the SAE threshold) a seeded 13,107 for RL and the rest for SFT
             -> /data/sae/feature_split_131k.npz (+ .json)
    build()  per split: candidates breadth-first over example ranks (up to k_sel per feature) -- window = the example's tokens cut to END
             at its peak token, leading special tokens stripped; dropped if a special token remains inside, < min_tok tokens, the
             decode -> re-encode roundtrip is not exact, or the text repeats within the feature. END-ANCHOR (GPU): every candidate
             re-forwarded standalone [sink]+ids through Qwen3.6-27B, kept iff the feature peaks on the LAST token (these windows were
             cut from 512-token documents, so context-dependent peaks fail here). Then up to k_keep kept windows per feature (rank order)
             and at most max_windows overall (breadth-first). Rows: "sae131k" = unit(encoder row f), "sae131k_dec" = unit(decoder column f).
             -> /data/banks/<out_name>/{vecs.f32 [N, 5120], records.jsonl (line i == vec_idx i), build_stats.json, meta.json}

    MODAL_PROFILE=safety-sahan modal deploy data/modal_sae131k_bank.py
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
app = modal.App(os.environ.get("SAE131K_BANK_APP", "maemm-27b-sae131k-bank"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0", "huggingface_hub==1.27.0",
                 "tokenizers==0.22.2", "hf_xet")
    .pip_install("flash-linear-attention==0.5.2")      # Qwen3.6 GDN forward kernels
    .env({"HF_HOME": "/data/hf_cache", "HF_HUB_OFFLINE": "1"})
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
SAE = "/data/sae/ae.pt"
MAXACTS = "/data/sae/maxacts_fresh.pt"
SPLIT = "/data/sae/feature_split_131k.npz"
POOL_HELDOUT = "/data/pool_heldout/records.jsonl"


def _load_sae():
    import torch
    p = torch.load(SAE, map_location="cpu", weights_only=False)
    return p["encoder.weight"].float(), p["decoder.weight"].float().T, p["encoder.bias"].float(), p["b_dec"].float(), float(p["threshold"])


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=64 * 1024, timeout=1800)
def split(seed: int = 2026, n_rl: int = 13_107):
    import json
    import numpy as np
    import torch
    vol.reload()
    W_enc, _, _, _, thr = _load_sae()
    ev = sorted({int(json.loads(l)["feature"]) for l in open(POOL_HELDOUT) if json.loads(l)["family"] == "sae"})
    es = torch.load("/data/eval_universal_ho/eval_sets_heldout_v3.pt", map_location="cpu", weights_only=False)
    assert set(map(int, es["sae_feats"])) <= set(ev), "v3 eval features must be inside the held-out split"
    top = torch.load(MAXACTS, map_location="cpu", weights_only=False)["max_acts"].amax(dim=(1, 2)).numpy()
    live = np.flatnonzero(top > thr)
    rest = np.setdiff1d(live, np.array(ev))
    perm = np.random.default_rng(seed).permutation(rest)
    sp = {"eval": np.array(ev, np.int64), "rl": np.sort(perm[:n_rl]), "sft": np.sort(perm[n_rl:])}
    np.savez(SPLIT, **sp)
    info = {"seed": seed, "n_features": int(W_enc.shape[0]), "threshold": thr, "live": int(len(live)), "eval_from": POOL_HELDOUT,
            "eval_live": int(np.isin(sp["eval"], live).sum()), **{k: int(len(v)) for k, v in sp.items()}}
    json.dump(info, open(SPLIT.replace(".npz", ".json"), "w"), indent=1)
    vol.commit()
    print(json.dumps(info), flush=True)
    return info


@app.function(image=image, gpu="H200:1", volumes={"/data": vol}, cpu=16, memory=192 * 1024, timeout=8 * 3600)
def build(split_key: str, k_keep: int, out_name: str, k_sel: int = 16, max_windows: int = 0, min_tok: int = 8, batch: int = 256,
          seed: int = 2040, overwrite: bool = False):
    import json, sys, time
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, "/pmx/helpers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mxf.config import D_MODEL, MODEL, PROFILE, READ_LAYER
    from mxf.inject import read_resid
    assert PROFILE == "qwen36-27b", PROFILE
    vol.reload()
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} already built")
    t0 = time.time()
    feats = np.load(SPLIT)[split_key].astype(np.int64)
    W_enc, W_dec, b_enc, b_dec, thr = _load_sae()
    d = torch.load(MAXACTS, map_location="cpu", weights_only=False)
    toks, acts = d["max_tokens"].long(), d["max_acts"].float()
    tok = AutoTokenizer.from_pretrained(MODEL)
    special = set(tok.all_special_ids) | set(int(i) for i in tok.added_tokens_decoder)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    # ---- 1. candidates (CPU): breadth-first over example ranks, up to k_sel per feature ----
    n_ex = toks.shape[1]
    per_feat = {int(f): [] for f in feats}; seen = {int(f): set() for f in feats}
    drop = {"zero_peak": 0, "special_inside": 0, "short": 0, "roundtrip": 0, "dup_text": 0}
    for r in range(n_ex):
        for f in feats:
            f = int(f)
            if len(per_feat[f]) >= k_sel:
                continue
            a = acts[f, r]; p = int(a.argmax())
            if float(a[p]) <= 0:
                drop["zero_peak"] += 1; continue
            ids = toks[f, r, : p + 1].tolist()
            while ids and ids[0] in special:
                ids.pop(0)
            if any(t in special for t in ids):
                drop["special_inside"] += 1; continue
            if len(ids) < min_tok:
                drop["short"] += 1; continue
            text = tok.decode(ids)
            if tok(text, add_special_tokens=False)["input_ids"] != ids:
                drop["roundtrip"] += 1; continue
            if text in seen[f]:
                drop["dup_text"] += 1; continue
            seen[f].add(text); per_feat[f].append((r, ids, text, float(a[p])))
    cand = [(f, *w) for f in per_feat for w in per_feat[f]]
    print(f"[sae131k] {split_key}: {len(feats)} features, {len(cand)} candidates (k_sel={k_sel}) in {time.time() - t0:.0f}s; drops {drop}", flush=True)

    # ---- 2. end-anchor (GPU): standalone re-forward through the truncated 27B, feature must peak on the last token ----
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
    # no truncation: Qwen3.6's hybrid layer_types are layer-indexed; read_resid stops at READ_LAYER anyway
    We = W_enc.cuda(); be = b_enc.cuda(); bd = b_dec.cuda()
    ok = np.zeros(len(cand), bool); pv = np.zeros(len(cand), np.float32)
    order = sorted(range(len(cand)), key=lambda i: len(cand[i][2]))
    for c0 in range(0, len(order), batch):
        idx = order[c0:c0 + batch]
        L = max(len(cand[i][2]) for i in idx) + 1
        ids = torch.full((len(idx), L), sink, dtype=torch.long); mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            w = [sink] + cand[i][2]; ids[j, : len(w)] = torch.tensor(w); mask[j, : len(w)] = 1
        with torch.no_grad():
            h, _ = read_resid(model, READ_LAYER, {"input_ids": ids.cuda(), "attention_mask": mask.cuda()}, pool="all")
            fsel = torch.tensor([cand[i][0] for i in idx], device="cuda")
            fa = torch.relu(torch.einsum("bld,bd->bl", h - bd, We[fsel]) + be[fsel][:, None])
            fa[:, 0] = -1.0; fa[mask.cuda() == 0] = -1.0
        last = (mask.sum(1) - 1).cuda(); am = fa.argmax(1)
        good = ((am == last) & (fa.gather(1, last[:, None]).squeeze(1) > 0)).cpu().numpy()
        vals = fa.gather(1, last[:, None]).squeeze(1).float().cpu().numpy()
        for j, i in enumerate(idx):
            ok[i] = good[j]; pv[i] = vals[j]
        if (c0 // batch) % 200 == 0:
            print(f"[sae131k] anchor {c0 + len(idx)}/{len(order)} kept so far {int(ok[order[:c0 + len(idx)]].sum())} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[sae131k] end-anchor: {int(ok.sum())} / {len(cand)} pass in {time.time() - t0:.0f}s", flush=True)

    # ---- 3. keep up to k_keep per feature (rank order), breadth-first under max_windows ----
    by_f = {}
    for i, (f, r, *_rest) in enumerate(cand):
        if ok[i]:
            by_f.setdefault(f, []).append(i)
    kept, level = [], 0
    while level < k_keep:
        added = [v[level] for v in by_f.values() if len(v) > level]
        if not added:
            break
        if max_windows and len(kept) + len(added) > max_windows:
            rng = np.random.default_rng(seed + level); added = list(rng.permutation(added)[: max_windows - len(kept)])
            kept += added; break
        kept += added; level += 1
    enc_u, dec_u = F.normalize(W_enc, dim=-1), F.normalize(W_dec, dim=-1)
    rows = []
    for i in kept:
        f, r, ids, text, corpus_act = cand[i]
        base = {"target_text": text, "feature": f, "rank": r, "n_tok": len(ids), "peak_act_standalone": round(float(pv[i]), 4),
                "peak_act_maxacts": round(corpus_act, 4), "src": "sae131k_maxacts_fresh_top16_ctx32"}
        rows.append(({**base, "family": "sae131k"}, ("enc", f))); rows.append(({**base, "family": "sae131k_dec"}, ("dec", f)))
    perm = np.random.default_rng(seed).permutation(len(rows))
    os.makedirs(out, exist_ok=True)
    V = np.memmap(f"{out}/vecs.f32.tmp", np.float32, "w+", shape=(len(rows), D_MODEL))
    with open(f"{out}/records.jsonl.tmp", "w") as fh:
        for vi, ri in enumerate(perm):
            rec, (kind, f) = rows[ri]
            V[vi] = (enc_u if kind == "enc" else dec_u)[f].numpy()
            fh.write(json.dumps({**rec, "vec_idx": vi}, ensure_ascii=False) + "\n")
    V.flush(); del V
    os.replace(f"{out}/vecs.f32.tmp", f"{out}/vecs.f32"); os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    wpf = np.bincount(np.array([cand[i][0] for i in kept], np.int64), minlength=W_enc.shape[0])[feats]
    stats = {"kind": "131k L42 SAE bank (in-distribution for the v3 'sae' eval family): unit encoder row (sae131k) / decoder column "
                     "(sae131k_dec) x end-anchored max-act window", "n_examples": len(rows), "families": {"sae131k": len(kept), "sae131k_dec": len(kept)},
             "split": split_key, "n_split_features": int(len(feats)), "features_with_rows": int((wpf > 0).sum()), "k_sel": k_sel, "k_keep": k_keep,
             "max_windows": max_windows, "windows_per_feature": {"mean": float(wpf.mean()), "median": float(np.median(wpf)), "max": int(wpf.max())},
             "candidates": len(cand), "anchor_pass": int(ok.sum()), "drops": drop, "threshold": thr, "min_tok": min_tok, "model": MODEL,
             "layer": READ_LAYER, "wall_s": round(time.time() - t0)}
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1); json.dump(stats, open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(stats), flush=True)
    return stats
