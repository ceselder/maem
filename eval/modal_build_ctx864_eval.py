"""Modal app (CPU): a SMALL eval-set cache that scores SFT-distribution real activations -- standalone 8-64-token windows,
the exact recipe of the simple2m SFT bank (ufw_ctx8_64_sft_4m_v3: collect --w-full, p 8-64) -- but from HELD-OUT documents
(Ultra-FineWeb eval head, doc_idx < 100,000; the SFT bank used docs [5.5M, 5.7M), the SAE corpus starts at 100k).

    <out> = copy of eval_sets_heldout_v3.pt with
        realact_ctx8_64_dirs [n, d]   unit(act - whiten_mu) rows of the held-out bank, at most one row per document first
        meta["cos_families"] = ["realact", "realact_ctx8_64"]    (realact kept as a sanity check vs the v3 numbers)
        meta["extra_families"] / meta["sae_slice_families"] removed  (skip the mlp + 2M-SAE extras -> ~1/6 of a full eval)
        meta["ctx864"] = provenance (bank, seed, doc range, ctx_len histogram)
    meta n / seed / heldout_pool / d_sae are unchanged, so the evaluator's held-out cache checks pass; the sae (131k) family
    always runs and stays comparable.

    MODAL_PROFILE=safety-sahan modal deploy eval/modal_build_ctx864_eval.py
    python -c "import modal; print(modal.Function.from_name('maemm-eval-cache-ctx864-s2m','build').remote())"
"""
import os

import modal

app = modal.App(os.environ.get("EVAL_CTX864_APP", "maemm-eval-cache-ctx864-s2m"))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy==2.4.6"))
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

V3 = "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
BANK = "/data/banks/ufw_ctx8_64_evalhead_32k"
OUT = "/data/eval_universal_ho/eval_sets_ctx864.pt"
FAM = "realact_ctx8_64"
EVAL_HEAD_DOCS = 100_000
D_MODEL = 5120


@app.function(image=image, cpu=8, memory=64 * 1024, volumes={"/data": vol}, timeout=3600)
def build(bank: str = BANK, n: int = 512, seed: int = 864, out: str = OUT, overwrite: bool = False):
    import json
    from collections import Counter
    import numpy as np
    import torch
    import torch.nn.functional as F
    vol.reload()
    if os.path.exists(out) and not overwrite:
        raise RuntimeError(f"{out} exists (overwrite=False)")
    stats = json.load(open(f"{bank}/build_stats.json"))
    recs = [json.loads(l) for l in open(f"{bank}/records.jsonl")]
    total = len(recs)
    vecs = np.memmap(f"{bank}/vecs.f16", np.float16, "r", shape=(total, D_MODEL))
    docs = np.array([r["doc_idx"] for r in recs])
    ctx = np.array([r["ctx_len"] for r in recs])
    assert docs.min() >= 0 and docs.max() < EVAL_HEAD_DOCS, f"doc_idx range [{docs.min()}, {docs.max()}] leaves the eval head"
    assert all(r.get("full_ctx") for r in recs) and ctx.min() >= 8 and ctx.max() <= 64, "not the SFT --w-full 8-64 recipe"
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    seen, first, rest = set(), [], []
    for i in order:                                   # one row per document first, then fill from the remainder
        (rest if docs[i] in seen else first).append(int(i)); seen.add(docs[i])
    sel = (first + rest)[:n]
    vidx = np.array([recs[i]["vec_idx"] for i in sel], np.int64)
    dirs = F.normalize(torch.from_numpy(np.asarray(vecs[vidx]).astype(np.float32)), dim=-1)
    es = torch.load(V3, map_location="cpu", weights_only=False)
    es[f"{FAM}_dirs"] = dirs
    meta = es["meta"]
    for k in ("extra_families", "sae_slice_families"):
        meta.pop(k, None)
    meta["cos_families"] = ["realact", FAM]
    meta["ctx864"] = {"bank": bank, "bank_rows": total, "seed": seed, "n": len(sel), "distinct_docs": len({docs[i] for i in sel}),
                      "doc_idx_range": [int(docs.min()), int(docs.max())], "ctx_len_hist": dict(Counter(int(ctx[i]) // 8 * 8 for i in sel)),
                      "rows": vidx.tolist(), "bank_stats_kind": stats.get("kind", "")}
    torch.save(es, out + ".tmp"); os.replace(out + ".tmp", out)
    vol.commit()
    info = {k: v for k, v in meta["ctx864"].items() if k != "rows"}
    print(json.dumps(info), flush=True)
    return info
