"""Modal app (CPU): the held-out eval-set cache for the Qwen3-8B port (same families / protocol as the 27B figures).

    /data/eval_universal_ho/eval_sets_8b.pt
        realact_ctx8_64_dirs     [512, 4096]  standalone 8-64-token windows (the SFT recipe), Ultra-FineWeb eval head (docs < 100k)
        realact_ctx64_2048_dirs  [512, 4096]  full-document forwards, ctx 64-2048 (the RL recipe), eval head (docs 50k-100k)
        random_dirs              [512, 4096]  seeded Gaussian control (never trained on)
        sae_dirs / sae_feats / corpus_peak    the UNSEEN SAE (the 27B's 131k role): 512 live features of Karvonen's 16k L27 SAE
                                              (trainer_0, never in any bank), unit encoder rows, peaks of our 60M-token ctx-32 scan;
                                              the always-scored "sae" family -> the evaluator's --eval-sae is /data/qwen3_8b/sae16k/ae.pt
        sae8b_{enc,dec}_dirs / _feats / _corpus_peak   the TRAINED SAE (the 27B's 2M role): 512 EVAL-split features of Karvonen's 65k
                                              L27 SAE (trainer_2) as encoder / decoder directions, peaks of his 60M-token max-acts scan,
                                              scored through the encoder SLICE (eval_universal.SliceSAE) exactly like sae2m_{enc,dec}
        sae_slice = {feats, W_enc [d, m], b_enc, b_dec, threshold, F, ae}
        meta: cos_families = [realact_ctx8_64, realact_ctx64_2048, random], sae_slice_families = [sae8b_enc, sae8b_dec], n, seed, d_sae

    MODAL_PROFILE=safety-sahan EVAL8B_APP=maemm-8b-eval-cache modal deploy eval/modal_build_eval_8b.py
    python -c "import modal; print(modal.Function.from_name('maemm-8b-eval-cache','build').remote())"
"""
import os

import modal

app = modal.App(os.environ.get("EVAL8B_APP", "maemm-8b-eval-cache"))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy==2.4.6"))
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
OUT = "/data/eval_universal_ho/eval_sets_8b.pt"
BANKS = {"realact_ctx8_64": "/data/banks/ufw8b_ctx8_64_evalhead_32k", "realact_ctx64_2048": "/data/banks/ufw8b_ctx64_2048_evalhead_32k"}
SAE = "/data/qwen3_8b/sae/ae.pt"                   # trained SAE (65k, trainer_2): slice families
SAE16K = "/data/qwen3_8b/sae16k/ae.pt"             # unseen SAE (16k, trainer_0): the "sae" family
PEAKS16K = "/data/qwen3_8b/sae16k/peaks.npy"
MAXACTS = "/data/qwen3_8b/sae/max_acts_top30_ctx32.pt"
SPLIT = "/data/qwen3_8b/sae/feature_split.npz"
D = 4096
EVAL_HEAD_DOCS = 100_000


def _pick(bank, n, seed):
    """n rows of a finalized realact bank, one per document first (then the remainder), all from the eval head."""
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    recs = [json.loads(l) for l in open(f"{bank}/records.jsonl")]
    vecs = np.memmap(f"{bank}/vecs.f16", np.float16, "r", shape=(len(recs), D))
    docs = np.array([r["doc_idx"] for r in recs])
    assert docs.min() >= 0 and docs.max() < EVAL_HEAD_DOCS, f"{bank}: doc_idx [{docs.min()}, {docs.max()}] leaves the eval head"
    order = np.random.default_rng(seed).permutation(len(recs))
    seen, first, rest = set(), [], []
    for i in order:
        (rest if docs[i] in seen else first).append(int(i)); seen.add(docs[i])
    sel = (first + rest)[:n]
    vidx = np.array([recs[i]["vec_idx"] for i in sel], np.int64)
    ctx = [int(recs[i].get("ctx_len", -1)) for i in sel]
    return F.normalize(torch.from_numpy(np.asarray(vecs[vidx]).astype(np.float32)), dim=-1), {
        "bank": bank, "rows": vidx.tolist(), "distinct_docs": len({docs[i] for i in sel}), "doc_range": [int(docs.min()), int(docs.max())],
        "ctx_len_median": float(np.median(ctx))}


@app.function(image=image, cpu=8, memory=64 * 1024, volumes={"/data": vol}, timeout=3600)
def build(n: int = 512, seed: int = 8, out: str = OUT, overwrite: bool = False):
    import json
    import numpy as np
    import torch
    import torch.nn.functional as F
    vol.reload()
    if os.path.exists(out) and not overwrite:
        raise RuntimeError(f"{out} exists")
    es, prov = {}, {}
    for fam, bank in BANKS.items():
        es[f"{fam}_dirs"], prov[fam] = _pick(bank, n, seed)
    g = torch.Generator().manual_seed(seed)
    es["random_dirs"] = F.normalize(torch.randn(n, D, generator=g), dim=-1)
    p = torch.load(SAE, map_location="cpu", weights_only=False)
    W_enc, W_dec = p["encoder.weight"].float(), p["decoder.weight"].float().T      # both [F, d]
    b_enc, b_dec, thr = p["encoder.bias"].float(), p["b_dec"].float(), float(p["threshold"])
    ev = np.load(SPLIT)["eval"]
    feats = np.sort(np.random.default_rng(seed).choice(ev, size=n, replace=False)).astype(np.int64)
    ma = torch.load(MAXACTS, map_location="cpu", weights_only=False)["max_acts"].float()
    peak = ma.amax(dim=(1, 2))[torch.from_numpy(feats)]
    ft = torch.from_numpy(feats)
    enc_u, dec_u = F.normalize(W_enc[ft], dim=-1), F.normalize(W_dec[ft], dim=-1)
    q = torch.load(SAE16K, map_location="cpu", weights_only=False)
    W16, thr16 = q["encoder.weight"].float(), float(q["threshold"])
    pk16 = np.load(PEAKS16K)
    live16 = np.flatnonzero(pk16 > thr16)
    f16 = np.sort(np.random.default_rng(seed + 1).choice(live16, size=n, replace=False)).astype(np.int64)
    es.update({"sae_dirs": F.normalize(W16[torch.from_numpy(f16)], dim=-1), "sae_feats": f16.tolist(),
               "corpus_peak": torch.from_numpy(pk16[f16]).float()})
    for fam, dirs in (("sae8b_enc", enc_u), ("sae8b_dec", dec_u)):
        es.update({f"{fam}_dirs": dirs, f"{fam}_feats": torch.from_numpy(feats), f"{fam}_corpus_peak": peak})
    es["sae_slice"] = {"feats": torch.from_numpy(feats), "W_enc": W_enc[ft].T.contiguous(), "b_enc": b_enc[ft], "b_dec": b_dec,
                       "threshold": thr, "F": int(W_enc.shape[0]), "ae": SAE, "maxacts": MAXACTS}
    es["meta"] = {"mode": "heldout_pool", "heldout_pool": "qwen3-8b eval head + Karvonen SAE eval split", "n": n, "seed": seed,
                  "d_sae": int(W16.shape[0]), "unseen_sae": {"path": SAE16K, "live": int(len(live16)), "threshold": thr16},
                  "cos_families": ["realact_ctx8_64", "realact_ctx64_2048", "random"],
                  "sae_slice_families": ["sae8b_enc", "sae8b_dec"], "version": "8b-1", "model": "Qwen/Qwen3-8B", "layer": 27,
                  "realact": prov, "sae_eval_features": {"n_eval_split": int(len(ev)), "threshold": thr, "corpus_peak_median": float(peak.median())}}
    torch.save(es, out + ".tmp"); os.replace(out + ".tmp", out)
    vol.commit()
    info = {**{k: v for k, v in es["meta"].items() if k != "realact"}, "realact": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in prov.items()}}
    print(json.dumps(info), flush=True)
    return info
