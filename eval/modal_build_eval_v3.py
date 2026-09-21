"""Modal app (CPU): eval-set cache v3 = the v2 held-out cache + two SAE-SLICE families for the 2,097,152-feature layer-42 SAE.

    /data/eval_universal_ho/eval_sets_heldout_v3.pt  = torch.load(v2) plus
        sae2m_enc_dirs [n, d]   unit ENCODER columns  unit(W_enc[:, f])   of n held-out 2M-SAE features
        sae2m_dec_dirs [n, d]   unit DECODER rows     unit(W_dec[f])      of the SAME features
        sae2m_{enc,dec}_feats [n] (feature ids), sae2m_{enc,dec}_corpus_peak [n] (top act over the 1.0B-token max-acts scan)
        sae_slice = {feats [n] sorted, W_enc [d, n], b_enc [n], b_dec [d], threshold, F, ae, maxacts}   (eval_universal.SliceSAE)
        meta["sae_slice_families"] = ["sae2m_enc", "sae2m_dec"], meta["version"] = 3, meta["v3"] = provenance
    Every v2 key (11 cos families, sae 131k family, mlp extras, meta.cos_families / n / seed / heldout_pool / d_sae) is UNCHANGED,
    so eval/mean_all stays comparable with every v2 run; the new families are logged as eval/sae2m_{enc,dec}/... only.

The n features are a seeded sample of the EVAL split of /data/sae2m/feature_split.npz (100,000 ids never in any training or RL
bank of the simple2m chain). Directions come straight from ae.pt (bf16 -> fp32, unit-normalised); the slice holds only their
encoder columns, so an evaluator never loads the 43 GB dictionary.

    MODAL_PROFILE=safety-sahan modal deploy eval/modal_build_eval_v3.py
    python -c "import modal; print(modal.Function.from_name('maemm-eval-cache-v3-s2m','build').remote())"
"""
import os

import modal

app = modal.App(os.environ.get("EVAL_V3_APP", "maemm-eval-cache-v3-s2m"))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy==2.4.6"))
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
# Deploy-time knobs (2026-09-21, the v2 SAE rerun): EVAL_V3_OUT / EVAL_V3_AE / EVAL_V3_MAXACTS / EVAL_V3_SPLIT point a fresh deployment
# (EVAL_V3_APP=maemm-eval-cache-v4-s2m) at another SAE's files; defaults = the original v3 build.
V3 = os.environ.get("EVAL_V3_OUT", "/data/eval_universal_ho/eval_sets_heldout_v3.pt")
AE = os.environ.get("EVAL_V3_AE", "/data/sae2m/trainer_0/ae.pt")
MAXACTS = os.environ.get("EVAL_V3_MAXACTS", "/data/sae2m/maxacts_top5.pt")
SPLIT = os.environ.get("EVAL_V3_SPLIT", "/data/sae2m/feature_split.npz")
FAM_ENC, FAM_DEC = "sae2m_enc", "sae2m_dec"


@app.function(image=image, cpu=16, memory=128 * 1024, volumes={"/data": vol}, timeout=3 * 3600)
def build(n: int = 512, seed: int = 2026, out: str = V3, overwrite: bool = False):
    import hashlib, json, time
    import numpy as np
    import torch
    import torch.nn.functional as F
    t0 = time.time()
    vol.reload()
    if os.path.exists(out) and not overwrite:
        raise RuntimeError(f"{out} exists (overwrite=False)")
    for p in (V2, AE, MAXACTS, SPLIT):
        assert os.path.exists(p), f"missing {p}"
    es = torch.load(V2, map_location="cpu", weights_only=False)
    assert "sae_slice_families" not in es["meta"], "v2 cache already has slice families?"
    n_v2 = int(es["meta"]["n"]); assert n <= n_v2 or True
    sp = np.load(SPLIT)
    ev_ids = np.asarray(sp["eval"], np.int64)
    for k in ("rl", "sft"):
        assert np.intersect1d(ev_ids, np.asarray(sp[k], np.int64)).size == 0, f"eval split overlaps {k}"
    rng = np.random.default_rng(seed)
    feats = np.sort(rng.choice(ev_ids, size=n, replace=False)).astype(np.int64)
    print(f"[v3] {n} eval features sampled from the {len(ev_ids)}-feature eval split (seed {seed}); first {feats[:6].tolist()}", flush=True)

    sd = torch.load(AE, map_location="cpu", mmap=True, weights_only=False)
    Fd, d = sd["encoder.weight"].shape                                   # nn.Linear [F, d]
    assert tuple(sd["decoder.weight"].shape) == (d, Fd), sd["decoder.weight"].shape
    ft = torch.from_numpy(feats)
    W_enc_cols = sd["encoder.weight"][ft].float()                        # [n, d]  (== W_enc[:, f] of mxf.sae)
    W_dec_rows = sd["decoder.weight"][:, ft].float().T.contiguous()       # [n, d]  (decoder column f = the feature's write direction)
    b_enc = sd["encoder.bias"][ft].float()
    b_dec = (sd["b_dec"] if "b_dec" in sd else sd["bias"]).float()
    thr = float(sd["threshold"].item())
    print(f"[v3] ae.pt F={Fd} d={d} thr={thr:.4f}; enc/dec slices read ({time.time() - t0:.0f}s)", flush=True)
    enc_dirs = F.normalize(W_enc_cols, dim=-1); dec_dirs = F.normalize(W_dec_rows, dim=-1)
    enc_dec_cos = (enc_dirs * dec_dirs).sum(-1)

    ma = torch.load(MAXACTS, map_location="cpu", weights_only=False)
    max_acts = ma["max_acts"].float()                                     # [F, N] (-1 empty)
    assert max_acts.shape[0] == Fd
    cp = max_acts[ft].max(1).values
    fire = ma["fire_counts"][ft]
    assert bool((cp > thr).all()), f"{int((cp <= thr).sum())} eval features never fire above the gate in the scan"
    print(f"[v3] corpus peaks: min {cp.min():.2f} median {cp.median():.2f} max {cp.max():.2f}; fire counts min {int(fire.min())} "
          f"median {float(fire.float().median()):.0f}; cos(enc, dec) median {enc_dec_cos.median():.3f} ({time.time() - t0:.0f}s)", flush=True)

    for fam, dirs in ((FAM_ENC, enc_dirs), (FAM_DEC, dec_dirs)):
        es[f"{fam}_dirs"] = dirs.contiguous()
        es[f"{fam}_feats"] = feats.tolist()
        es[f"{fam}_corpus_peak"] = cp.clone()
    es["sae_slice"] = {"feats": ft.clone(), "W_enc": W_enc_cols.T.contiguous(), "b_enc": b_enc.clone(), "b_dec": b_dec.clone(),
                       "threshold": thr, "F": int(Fd), "ae": AE, "maxacts": MAXACTS, "k": int(sd["k"].item()) if "k" in sd else None}
    es["meta"]["sae_slice_families"] = [FAM_ENC, FAM_DEC]
    es["meta"]["version"] = 3
    es["meta"]["v3"] = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from": V2, "n": int(n), "seed": int(seed),
                        "feature_split": SPLIT, "split_key": "eval", "n_eval_split": int(len(ev_ids)),
                        "split_sha256_16": hashlib.sha256(open(SPLIT, "rb").read()).hexdigest()[:16],
                        "ae": AE, "F": int(Fd), "threshold": thr, "maxacts": MAXACTS, "maxacts_tokens_seen": int(ma.get("tokens_seen", 0) or 0),
                        "dirs": {FAM_ENC: "unit(W_enc[:, f]) = unit(encoder.weight[f])", FAM_DEC: "unit(W_dec[f]) = unit(decoder.weight[:, f])"},
                        "corpus_peak": "max over the top-5 stored activations of the 1.0B-token Ultra-FineWeb max-acts scan",
                        "enc_dec_cos": {"median": float(enc_dec_cos.median()), "min": float(enc_dec_cos.min()), "max": float(enc_dec_cos.max())},
                        "note": "v2 keys untouched -> eval/mean_all comparable; slice families are logged as eval/<fam>/* only"}
    tmp = out + ".tmp"
    torch.save(es, tmp); os.replace(tmp, out); vol.commit()
    chk = torch.load(out, map_location="cpu", weights_only=False)
    assert chk["meta"]["sae_slice_families"] == [FAM_ENC, FAM_DEC] and chk[f"{FAM_ENC}_dirs"].shape == (n, d)
    summary = {"out": out, "bytes": os.path.getsize(out), "n": n, "F": int(Fd), "threshold": thr, "families": chk["meta"]["cos_families"],
               "extra": chk["meta"].get("extra_families"), "slice": chk["meta"]["sae_slice_families"], "corpus_peak_median": float(cp.median()),
               "enc_dec_cos_median": float(enc_dec_cos.median()), "wall_s": time.time() - t0}
    print(json.dumps(summary), flush=True)
    return summary
