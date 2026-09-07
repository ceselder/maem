"""Modal app `maemm-mlp42-bank`: layer-42 MLP neurons (singles + co-firing pair composites) as the SIXTH inverter direction
family — bank /data/banks/mlp42, eval cache v2, and the merged training mix /data/banks/mix_1m_mlp.
Logic lives in data/mlp42_bank_worker.py (mounted at /pmx/helpers); see its docstring for every definition.

Stages (profile safety-sahan; deploy + spawn survives the launching client):
    modal deploy data/modal_mlp42_bank.py
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-bank', 'scan').spawn().object_id)"      # 1 GPU, ~10 min
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-bank', 'build').spawn().object_id)"     # small GPU, ~10 min
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-bank', 'merge').spawn().object_id)"     # CPU, ~10 min
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-bank', 'peek').remote())"
Outputs: /data/mlp42/bank_scan.npz, /data/mlp42/bank_selection.json, /data/banks/mlp42/{vecs.f32,records.jsonl,build_stats.json,
meta.json}, /data/eval_universal_ho/eval_sets_heldout_v2.pt (the v1 cache is never touched), /data/banks/mix_1m_mlp/.

EXPANDED bank (>= 250k rows for a bigger SFT mix; same split, nothing existing overwritten, eval cache read-only):
    f = modal.Function.from_name('maemm-mlp42-bank', 'scan'); f.spawn(n_windows=80000, batch=32, topk=40, sample_seed=2027,
        sel_file='/data/mlp42/sel_windows_big.npz', scan_file='/data/mlp42/bank_scan_big.npz')          # 1 GPU, ~1 h
    f = modal.Function.from_name('maemm-mlp42-bank', 'build'); f.spawn(k_single=32, k_pair=8, check_mix=False, min_c=500,
        scan_file='/data/mlp42/bank_scan_big.npz', bank_out='/data/banks/mlp42_big', write_eval_cache=False,
        selection_file='/data/mlp42/bank_selection_big.json')     # min_c 500 = 10 * 20.48M/409.6k: same joint-firing RATE floor
    modal.Function.from_name('maemm-mlp42-bank', 'verify').remote(bank='/data/banks/mlp42_big')
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = "maemm-mlp42-bank"
app = modal.App(APP_NAME)

# same pins as data/modal_mlp42_neurons.py (one environment across the suite)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet", "scipy==1.17.1")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "data" / "mlp42_neurons_worker.py", "/pmx/helpers/mlp42_neurons_worker.py")
    .add_local_file(REPO / "data" / "mlp42_bank_worker.py", "/pmx/helpers/mlp42_bank_worker.py")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
SCAN_GPUS = ["B200", "H200"]                          # forward-only: whichever schedules first
SMALL_GPUS = ["H100", "A100-80GB", "L40S", "A100-40GB"]   # leak check only (a few GB)


def _env():
    import os
    import sys
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, "/pmx/helpers")


@app.function(image=image, gpu=SCAN_GPUS, cpu=8, memory=65536, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=6 * 3600)
def scan(n_windows: int = 1600, win_len: int = 256, batch: int = 16, topk: int = 32, sample_seed: int | None = None,
         sel_file: str | None = None, scan_file: str | None = None):
    """Defaults == today's bank (first n_windows of sel_windows.npz -> bank_scan.npz). For an EXPANDED scan pass sample_seed
    (fresh windows over all acts27b TRAIN rows) + NEW sel_file/scan_file paths; nothing existing is overwritten."""
    _env()
    import time
    import torch
    from transformers import AutoModelForCausalLM
    import mlp42_bank_worker as BW
    from mxf.config import MODEL
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    vol.reload()
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": dev})
    base.eval()
    BW.log(f"base loaded in {time.time() - t0:.0f}s")
    res = BW.run_scan(base, n_windows=n_windows, win_len=win_len, batch=batch, topk=topk, dev=dev, sample_seed=sample_seed,
                      sel_file=sel_file, scan_file=scan_file)
    vol.commit()
    return res


@app.function(image=image, gpu=SMALL_GPUS, cpu=8, memory=65536, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=2 * 3600)
def build(seed: int = 2026, heldout_frac: float = 0.10, n_eval_single: int = 512, n_eval_pair: int = 256, k_single: int = 8,
          k_pair: int = 4, w_lo: int = 16, w_hi: int = 32, min_tok: int = 8, check_mix: bool = True, scan_file: str | None = None,
          bank_out: str | None = None, write_eval_cache: bool = True, selection_file: str | None = None, min_c: int = 10,
          min_lift: float = 10.0, max_p: float = 1e-10):
    """Defaults == today's bank (draws the hold-out split, writes eval cache v2 -> /data/banks/mlp42). For an EXPANDED bank pass
    scan_file (a bank_scan_big.npz), bank_out (new dir), write_eval_cache=False (hold-out + eval dirs READ from the existing v2
    cache, nothing under /data/eval_universal_ho is written), selection_file (new path), k_single/k_pair, and min_c scaled with the
    scan's token count (min_c = round(10 * T / 409600), i.e. 500 at 20.48M tokens) so the pair rule keeps its semantics."""
    _env()
    from transformers import AutoTokenizer
    import mlp42_bank_worker as BW
    from mxf.config import MODEL
    vol.reload()
    tok = AutoTokenizer.from_pretrained(MODEL)
    res = BW.run_build(tok, dev="cuda:0", seed=seed, heldout_frac=heldout_frac, n_eval_single=n_eval_single, n_eval_pair=n_eval_pair,
                       k_single=k_single, k_pair=k_pair, w_lo=w_lo, w_hi=w_hi, min_tok=min_tok, check_mix=check_mix,
                       scan_file=scan_file, bank_out=bank_out or BW.BANK_OUT, write_eval_cache=write_eval_cache, selection_file=selection_file,
                       min_c=min_c, min_lift=min_lift, max_p=max_p)
    vol.commit()
    return res


@app.function(image=image, cpu=8, memory=98304, ephemeral_disk=512 * 1024, volumes={"/data": vol}, timeout=3 * 3600)
def merge(seed: int = 17):
    _env()
    import mlp42_bank_worker as BW
    vol.reload()
    res = BW.run_merge(seed=seed)
    vol.commit()
    return res


@app.function(image=image, gpu=SMALL_GPUS, cpu=8, memory=65536, volumes={"/data": vol}, timeout=3600)
def verify(bank: str = "/data/banks/mlp42_big", ref_bank: str = "/data/banks/mlp42"):
    """Read-only: format of `bank` == `ref_bank`, no held-out neuron / eval pair inside, max cos vs every v2 eval direction."""
    _env()
    import mlp42_bank_worker as BW
    vol.reload()
    return BW.run_verify(bank, ref_bank=ref_bank, dev="cuda:0")


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, timeout=1800)
def peek(bank: str = "/data/banks/mix_1m_mlp", n: int = 3, families: str = "mlp,mlp_pair"):
    """Print build_stats + n sample rows per requested family (with the unit-norm check of their vectors)."""
    _env()
    import json
    import os
    import numpy as np
    from mxf.config import D_MODEL
    vol.reload()
    st = json.load(open(f"{bank}/build_stats.json"))
    print(json.dumps({k: st[k] for k in st if k not in ("parts",)}, indent=1), flush=True)
    want = [f for f in families.split(",") if f]
    N = st["n_examples"]
    assert os.path.getsize(f"{bank}/vecs.f32") == N * D_MODEL * 4
    vecs = np.memmap(f"{bank}/vecs.f32", np.float32, "r", shape=(N, D_MODEL))
    seen = {f: 0 for f in want}
    out = []
    with open(f"{bank}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            f = r["family"]
            if f in seen and seen[f] < n:
                seen[f] += 1
                assert r["vec_idx"] == i
                v = np.asarray(vecs[i], dtype=np.float32)
                r2 = {k: v2 for k, v2 in r.items() if k != "target_text"}
                out.append({"row": i, "norm": float(np.linalg.norm(v)), "text": r["target_text"], "rec": r2})
                print(f"[{f}] row {i} |v|={np.linalg.norm(v):.4f} :: {r['target_text']!r}\n      {json.dumps(r2)}", flush=True)
            if all(c >= n for c in seen.values()):
                break
    return out


@app.local_entrypoint()
def main(stage: str = "peek"):
    if stage == "scan":
        print(scan.remote())
    elif stage == "build":
        print(build.remote())
    elif stage == "merge":
        print(merge.remote())
    else:
        peek.remote()
