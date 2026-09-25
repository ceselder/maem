"""Modal app `maemm-8b-sae-bank`: SAE-feature banks for the Qwen3-8B port of the simple2m chain, built from Adam Karvonen's layer-27
BatchTopK SAE (/data/qwen3_8b/sae/ae.pt: 65,536 features, k=80) and its max-activating examples
(/data/qwen3_8b/sae/max_acts_top30_ctx32.pt: tokens + activations [65536, 30, 32], 60M-token scan, 32-token contexts).
Same construction as the 2M-SAE banks of the 27B chain (sae2m_bank/modal_sae2m_bank.py):

    split()  live features (top max-act example above the SAE's learned threshold), seeded permutation -> eval 5% / rl 10% / sft 85%
             -> /data/qwen3_8b/sae/feature_split.npz (+ .json summary)
    build()  per split: breadth-first over example ranks (every feature gets its rank-0 window before any gets rank-1, ...), up to
             `k` windows per feature. Window = the example's tokens cut to END at its peak token (leading special tokens stripped);
             dropped if a special token remains inside, < min_tok tokens, the decode -> re-encode roundtrip is not exact, or the text
             repeats within the feature. END-ANCHOR (GPU): each window re-forwarded standalone [sink]+ids through Qwen3-8B, kept iff
             the feature's pre-topk activation relu((h-b_dec).W_enc[:,f]+b_enc[f]) peaks on the LAST token. Two rows per kept window
             with the same target_text: family "sae8b" = unit(encoder row f), "sae8b_dec" = unit(decoder column f), raw layer-27 space.
             -> /data/banks/<out_name>/{vecs.f32 [N, 4096], records.jsonl (line i == vec_idx i), build_stats.json, meta.json}

    MODAL_PROFILE=safety-sahan modal deploy data/modal_sae8b_bank.py
    python -c "import modal; print(modal.Function.from_name('maemm-8b-sae-bank','split').remote())"
    python -c "import modal; print(modal.Function.from_name('maemm-8b-sae-bank','build').remote(split_key='sft', k=9, out_name='sae8b_sft'))"
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
app = modal.App(os.environ.get("SAE8B_BANK_APP", "maemm-8b-sae-bank"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0", "huggingface_hub==1.27.0",
                 "tokenizers==0.22.2", "hf_xet")
    .env({"HF_HOME": "/data/hf_cache", "HF_HUB_OFFLINE": "1"})
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
SAE = "/data/qwen3_8b/sae/ae.pt"
MAXACTS = "/data/qwen3_8b/sae/max_acts_top30_ctx32.pt"
SPLIT = "/data/qwen3_8b/sae/feature_split.npz"
FIRST_SPECIAL = 151643          # Qwen3 vocab: every id >= 151643 is a special / control token


def _load_sae():
    import torch
    p = torch.load(SAE, map_location="cpu", weights_only=False)
    W_enc = p["encoder.weight"].float()            # [F, d]  row f = encoder direction of feature f
    W_dec = p["decoder.weight"].float().T          # [F, d]  row f = decoder direction of feature f
    return W_enc, W_dec, p["encoder.bias"].float(), p["b_dec"].float(), float(p["threshold"])


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=64 * 1024, timeout=1800)
def split(seed: int = 2026, frac_eval: float = 0.05, frac_rl: float = 0.10):
    import json
    import numpy as np
    import torch
    vol.reload()
    W_enc, _, _, _, thr = _load_sae()
    ma = torch.load(MAXACTS, map_location="cpu", weights_only=False)["max_acts"]
    top = ma.float().amax(dim=(1, 2)).numpy()
    live = np.flatnonzero(top > thr)
    perm = np.random.default_rng(seed).permutation(live)
    n_ev, n_rl = int(round(frac_eval * len(live))), int(round(frac_rl * len(live)))
    sp = {"eval": np.sort(perm[:n_ev]), "rl": np.sort(perm[n_ev:n_ev + n_rl]), "sft": np.sort(perm[n_ev + n_rl:])}
    np.savez(SPLIT, **sp)
    info = {"seed": seed, "n_features": int(W_enc.shape[0]), "threshold": thr, "live": int(len(live)), **{k: int(len(v)) for k, v in sp.items()}}
    json.dump(info, open(SPLIT.replace(".npz", ".json"), "w"), indent=1)
    vol.commit()
    print(json.dumps(info), flush=True)
    return info


@app.function(image=image, gpu="H200:1", volumes={"/data": vol}, cpu=16, memory=128 * 1024, timeout=4 * 3600)
def build(split_key: str, k: int, out_name: str, min_tok: int = 8, batch: int = 512, seed: int = 2040, overwrite: bool = False):
    import json, sys, time
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, "/pmx/helpers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mxf.config import D_MODEL, MODEL, PROFILE, READ_LAYER
    from mxf.inject import read_resid
    assert PROFILE == "qwen3-8b", PROFILE
    vol.reload()
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} already built")
    t0 = time.time()
    feats = np.load(SPLIT)[split_key].astype(np.int64)
    W_enc, W_dec, b_enc, b_dec, thr = _load_sae()
    d = torch.load(MAXACTS, map_location="cpu", weights_only=False)
    toks, acts = d["max_tokens"], d["max_acts"].float()
    tok = AutoTokenizer.from_pretrained(MODEL)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    # ---- 1. selection (CPU): breadth-first over example ranks ----
    n_ex = toks.shape[1]
    per_feat = {int(f): [] for f in feats}
    seen = {int(f): set() for f in feats}
    drop = {"zero_peak": 0, "special_inside": 0, "short": 0, "roundtrip": 0, "dup_text": 0}
    for r in range(n_ex):
        for f in feats:
            f = int(f)
            if len(per_feat[f]) >= k:
                continue
            a = acts[f, r]
            p = int(a.argmax())
            if float(a[p]) <= 0:
                drop["zero_peak"] += 1; continue
            ids = toks[f, r, : p + 1].tolist()
            while ids and ids[0] >= FIRST_SPECIAL:
                ids.pop(0)
            if any(t >= FIRST_SPECIAL for t in ids):
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
    print(f"[sae8b] {split_key}: {len(feats)} features, {len(cand)} candidate windows (k={k}) in {time.time() - t0:.0f}s; drops {drop}", flush=True)

    # ---- 2. end-anchor (GPU): standalone re-forward, feature must peak on the last token ----
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    Wd = W_enc.cuda(); be = b_enc.cuda(); bd = b_dec.cuda()
    kept, anchor_fail, peak_vals = [], 0, []
    order = sorted(range(len(cand)), key=lambda i: len(cand[i][2]))       # length-sorted batches -> little padding
    for c0 in range(0, len(order), batch):
        idx = order[c0:c0 + batch]
        L = max(len(cand[i][2]) for i in idx) + 1
        ids = torch.full((len(idx), L), sink, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            w = [sink] + cand[i][2]
            ids[j, : len(w)] = torch.tensor(w); mask[j, : len(w)] = 1
        with torch.no_grad():
            h, _ = read_resid(model, READ_LAYER, {"input_ids": ids.cuda(), "attention_mask": mask.cuda()}, pool="all")   # [B, L, d]
            fsel = torch.tensor([cand[i][0] for i in idx], device="cuda")
            pre = torch.einsum("bld,bd->bl", h - bd, Wd[fsel]) + be[fsel][:, None]
            fa = torch.relu(pre)
            fa[:, 0] = -1.0                                                 # sink position never counts
            fa[mask.cuda() == 0] = -1.0
        last = mask.sum(1) - 1
        am = fa.argmax(1).cpu()
        for j, i in enumerate(idx):
            if int(am[j]) == int(last[j]) and float(fa[j, last[j]]) > 0:
                kept.append(i); peak_vals.append(float(fa[j, last[j]]))
            else:
                anchor_fail += 1
    print(f"[sae8b] end-anchor: kept {len(kept)} / {len(cand)} ({anchor_fail} failed) in {time.time() - t0:.0f}s", flush=True)

    # ---- 3. rows: encoder + decoder direction per kept window, seeded global shuffle ----
    enc_u = F.normalize(W_enc, dim=-1); dec_u = F.normalize(W_dec, dim=-1)
    rows = []
    for i, pv in zip(kept, peak_vals):
        f, r, ids, text, corpus_act = cand[i]
        base = {"target_text": text, "feature": f, "rank": r, "n_tok": len(ids), "peak_act_standalone": round(pv, 4),
                "peak_act_maxacts": round(corpus_act, 4), "src": "karvonen_max_acts_top30_ctx32"}
        rows.append(({**base, "family": "sae8b"}, ("enc", f)))
        rows.append(({**base, "family": "sae8b_dec"}, ("dec", f)))
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
    wpf = np.bincount(np.array([cand[i][0] for i in kept]), minlength=W_enc.shape[0])[feats]
    stats = {"kind": "Karvonen Qwen3-8B L27 SAE bank: unit encoder row (sae8b) / decoder column (sae8b_dec) x end-anchored max-act window",
             "n_examples": len(rows), "families": {"sae8b": len(kept), "sae8b_dec": len(kept)}, "split": split_key,
             "n_split_features": int(len(feats)), "features_with_rows": int((wpf > 0).sum()), "k": k,
             "windows_per_feature": {"mean": float(wpf.mean()), "median": float(np.median(wpf)), "min": int(wpf.min()), "max": int(wpf.max())},
             "drops": drop, "anchor_fail": anchor_fail, "threshold": thr, "min_tok": min_tok, "model": MODEL, "layer": READ_LAYER,
             "files": {"vecs.f32": f"float32 [{len(rows)},{D_MODEL}]", "records.jsonl": "line i == vec_idx i"}, "wall_s": round(time.time() - t0)}
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    json.dump(stats, open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    print(json.dumps({k2: v for k2, v in stats.items() if k2 != "files"}), flush=True)
    return stats
