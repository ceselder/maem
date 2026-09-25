"""Modal app `maemm-8b-prep`: one-time assets for the Qwen3-8B port of the simple2m chain (branch qwen3-8b, mxf/PROFILE = qwen3-8b).

    fetch_assets()  Qwen/Qwen3-8B into the volume HF cache (/data/hf_cache, refs/main set -> offline loads work) + Adam Karvonen's
                    layer-27 BatchTopK SAE (adamkarvonen/qwen3-8b-saes resid_post_layer_27/trainer_2: ae.pt, config.json,
                    eval_results.json) and its max-activating examples (adamkarvonen/sae_max_acts, 65,536 x 30 x 32 tokens)
                    -> /data/qwen3_8b/sae/
    whiten_mu()     the layer-27 corpus mean /data/qwen3_8b/whiten_mu_l27.npy, computed exactly like the 27B's acts27b/whiten_mu.npy:
                    [sink] + 512-token windows forwarded alone, mean over every CONTENT token (sink position dropped), no norm filter.
                    Documents: Ultra-FineWeb en, ordered stream from index 13,000,000 (disjoint from every training / eval range).

    MODAL_PROFILE=safety-sahan modal deploy data/modal_qwen3_8b_prep.py
    python -c "import modal; print(modal.Function.from_name('maemm-8b-prep','fetch_assets').remote())"
    python -c "import modal; print(modal.Function.from_name('maemm-8b-prep','whiten_mu').remote())"
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
app = modal.App(os.environ.get("PREP8B_APP", "maemm-8b-prep"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0", "huggingface_hub==1.27.0",
                 "tokenizers==0.22.2", "hf_xet", "datasets")
    .env({"HF_HOME": "/data/hf_cache"})
    .add_local_file(REPO / "data" / "collect_acts27b_worker.py", "/pmx/collect_acts27b_worker.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
SAE_DIR = "/data/qwen3_8b/sae"
SAE_SUB = "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_27/trainer_2"
MAXACTS = "acts_Qwen_Qwen3-8B_layer_27_trainer_2_layer_percent_75_context_length_32.pt"


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], cpu=8, memory=32 * 1024, timeout=3600)
def fetch_assets():
    import json, shutil
    from huggingface_hub import hf_hub_download, snapshot_download
    snap = snapshot_download("Qwen/Qwen3-8B")
    os.makedirs(SAE_DIR, exist_ok=True)
    out = {"model_snapshot": snap}
    for fn in ("ae.pt", "config.json", "eval_results.json"):
        p = hf_hub_download("adamkarvonen/qwen3-8b-saes", f"{SAE_SUB}/{fn}")
        shutil.copyfile(p, f"{SAE_DIR}/{fn}"); out[fn] = f"{SAE_DIR}/{fn}"
    p = hf_hub_download("adamkarvonen/sae_max_acts", MAXACTS, repo_type="dataset")
    shutil.copyfile(p, f"{SAE_DIR}/max_acts_top30_ctx32.pt"); out["max_acts"] = f"{SAE_DIR}/max_acts_top30_ctx32.pt"
    json.dump(out, open(f"{SAE_DIR}/SOURCES.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(out), flush=True)
    return out


@app.function(image=image, gpu="H200:1", volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], cpu=8,
              memory=64 * 1024, timeout=3 * 3600)
def whiten_mu(n_windows: int = 8192, skip: int = 13_000_000, batch: int = 32, max_wins_per_doc: int = 4,
              out: str = "/data/qwen3_8b/whiten_mu_l27.npy"):
    import json, sys, time
    import numpy as np
    import torch
    sys.path.insert(0, "/pmx"); sys.path.insert(0, "/pmx/helpers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collect_acts27b_worker import StreamReader
    from mxf.config import D_MODEL, MODEL, PROFILE, READ_LAYER
    from mxf.inject import read_resid
    assert PROFILE == "qwen3-8b", PROFILE
    L = 512
    tok = AutoTokenizer.from_pretrained(MODEL)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    model.model.layers = model.model.layers[: READ_LAYER + 1]           # truncated forward: nothing past the read layer is needed
    reader = StreamReader({"dataset": "openbmb/Ultra-FineWeb", "config": None, "split": "en", "skip": skip, "shuffle": False},
                          0, 1, 0, None)
    musum, count, n_done, wins, t0 = np.zeros(D_MODEL, np.float64), 0, 0, [], time.time()

    def flush(ws):
        nonlocal musum, count
        ids = torch.tensor([[sink] + w for w in ws], device="cuda:0")
        with torch.no_grad():
            h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
        content = h[:, 1:, :]
        musum += content.reshape(-1, D_MODEL).sum(0).double().cpu().numpy(); count += content.shape[0] * L

    for text in reader.docs():
        t = tok(text, add_special_tokens=False)["input_ids"]
        for k in range(min(max_wins_per_doc, len(t) // L)):
            wins.append(t[k * L:(k + 1) * L])
        while len(wins) >= batch:
            flush(wins[:batch]); wins = wins[batch:]; n_done += batch
        if n_done >= n_windows:
            break
    mu = (musum / count).astype(np.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, mu)
    meta = {"model": MODEL, "layer": READ_LAYER, "tokens": int(count), "windows": n_done, "window_len": L, "sink": int(sink),
            "doc_range": reader.state()["doc_range"], "mu_norm": float(np.linalg.norm(mu)), "seconds": round(time.time() - t0)}
    json.dump(meta, open(out.replace(".npy", ".json"), "w"), indent=1)
    vol.commit()
    print(json.dumps(meta), flush=True)
    return meta


SAE16K_DIR = "/data/qwen3_8b/sae16k"
SAE16K_SUB = "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_27/trainer_0"   # 16,384 features, k=80: the UNSEEN eval SAE of the 8B chain


@app.function(image=image, gpu="H200:1", volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], cpu=16,
              memory=64 * 1024, timeout=4 * 3600)
def sae16k_peaks(n_tokens: int = 60_000_000, skip: int = 14_000_000, ctx: int = 32, batch: int = 1024):
    """Karvonen's 16k L27 SAE (trainer_0; no published max-acts file) -> /data/qwen3_8b/sae16k/{ae.pt, config.json, peaks.npy}.
    peaks[f] = max pre-topk activation relu((h-b_dec).W_enc[:,f]+b_enc[f]) over an n_tokens scan of standalone ctx-token contexts
    ([sink] + ctx-1 text tokens, sink position excluded) -- the same context setup as his 65k max-acts file (60M tokens, ctx 32).
    Documents: Ultra-FineWeb en, ordered stream from index 14,000,000 (disjoint from every training / eval range)."""
    import json, shutil, sys, time
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, "/pmx"); sys.path.insert(0, "/pmx/helpers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collect_acts27b_worker import StreamReader
    from mxf.config import MODEL, PROFILE, READ_LAYER
    from mxf.inject import read_resid
    assert PROFILE == "qwen3-8b", PROFILE
    os.makedirs(SAE16K_DIR, exist_ok=True)
    for fn in ("ae.pt", "config.json", "eval_results.json"):
        shutil.copyfile(hf_hub_download("adamkarvonen/qwen3-8b-saes", f"{SAE16K_SUB}/{fn}"), f"{SAE16K_DIR}/{fn}")
    p = torch.load(f"{SAE16K_DIR}/ae.pt", map_location="cpu", weights_only=False)
    W = p["encoder.weight"].float().T.contiguous().cuda()       # [d, F]
    be, bd, thr = p["encoder.bias"].float().cuda(), p["b_dec"].float().cuda(), float(p["threshold"])
    tok = AutoTokenizer.from_pretrained(MODEL)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    reader = StreamReader({"dataset": "openbmb/Ultra-FineWeb", "config": None, "split": "en", "skip": skip, "shuffle": False}, 0, 1, 0, None)
    peaks = torch.zeros(W.shape[1], device="cuda"); fires = torch.zeros(W.shape[1], device="cuda", dtype=torch.long)
    L, seen, chunks, t0 = ctx - 1, 0, [], time.time()

    def run(cs):
        ids = torch.tensor([[sink] + c for c in cs], device="cuda:0")
        with torch.no_grad():
            h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
            a = torch.relu((h[:, 1:, :].reshape(-1, h.shape[-1]) - bd) @ W + be)
            peaks.copy_(torch.maximum(peaks, a.amax(0))); fires.add_((a > thr).sum(0))

    for text in reader.docs():
        t = tok(text, add_special_tokens=False)["input_ids"]
        chunks += [t[i:i + L] for i in range(0, len(t) - L + 1, L)]
        while len(chunks) >= batch:
            run(chunks[:batch]); chunks = chunks[batch:]; seen += batch * L
        if seen >= n_tokens:
            break
    np.save(f"{SAE16K_DIR}/peaks.npy", peaks.cpu().numpy()); np.save(f"{SAE16K_DIR}/fire_counts.npy", fires.cpu().numpy())
    meta = {"sae": f"{SAE16K_DIR}/ae.pt", "dict_size": int(W.shape[1]), "threshold": thr, "tokens": int(seen), "ctx": ctx,
            "doc_range": reader.state()["doc_range"], "live_features": int((peaks > thr).sum()), "peak_median": float(peaks.median()),
            "seconds": round(time.time() - t0)}
    json.dump(meta, open(f"{SAE16K_DIR}/peaks.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(meta), flush=True)
    return meta
