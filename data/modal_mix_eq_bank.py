"""Compose an EQUAL-SHARE 6-family SFT midtrain bank from two finalized source banks:
    /data/everything_eq240k   (bank_everything: realact / realact_long / sae / bsf / cluster; vecs.f32 unit rows + records.jsonl)
    /data/banks/mlp42_big     (expanded layer-42 MLP neuron bank: mlp / mlp_pair)
Rule (user): every family gets exactly --n-per-family rows; the MLP family is filled with individual neurons first and only
then with neuron pairs. Rows are a seeded global shuffle. Output (rl/rl.py + sft/pretrain.py bank format):
    /data/banks/<out>/vecs.f32 [N,5120], records.jsonl (line i == vec_idx i; keeps the source record + src_bank/src_vec_idx),
    build_stats.json {n_examples, families, ...}, meta.json.
Leakage: both sources were leak-checked against the eval cache / pool_heldout at build time; we re-assert max cos < 0.999
vs the eval-cache v2 directions as a cheap guard.
    modal deploy data/modal_mix_eq_bank.py && python -c "import modal; print(modal.Function.from_name('maemm-mix-eq-bank','build').spawn().object_id)"
"""
import json
import os

import modal

app = modal.App("maemm-mix-eq-bank")
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.2.6", "torch==2.8.0", extra_index_url="https://download.pytorch.org/whl/cpu")
D = 5120
SRC_5 = "/data/everything_eq240k"
SRC_MLP = "/data/banks/mlp42_big"
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
FAMILIES = ("realact", "realact_long", "sae", "bsf", "cluster", "mlp")   # mlp = mlp singles first, then mlp_pair


def _records(bank):
    with open(f"{bank}/records.jsonl") as f:
        recs = [json.loads(l) for l in f]
    n = os.path.getsize(f"{bank}/vecs.f32") // (4 * D)
    assert n == len(recs), f"{bank}: {n} vec rows vs {len(recs)} records"
    return recs, n


@app.function(image=image, cpu=16, memory=196_608, ephemeral_disk=256 * 1024, volumes={"/data": vol}, timeout=4 * 3600)
def build(out_name: str = "mix_eq_1p45m", n_per_family: int = 241_741, seed: int = 2028, overwrite: bool = False):
    import time
    import numpy as np
    import torch
    t0 = time.time()
    vol.reload()
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} already finalized (overwrite=False)")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    picks = []   # (src_bank, src_row, family_label)
    counts = {}
    for bank, fams in ((SRC_5, ("realact", "realact_long", "sae", "bsf", "cluster")), (SRC_MLP, ("mlp",))):
        recs, n = _records(bank)
        by = {}
        for i, r in enumerate(recs):
            by.setdefault(r["family"], []).append(i)
        for fam in fams:
            if fam == "mlp":
                singles = np.array(by.get("mlp", []), dtype=np.int64); pairs = np.array(by.get("mlp_pair", []), dtype=np.int64)
                take_s = rng.permutation(singles)[:n_per_family]
                take_p = rng.permutation(pairs)[: max(0, n_per_family - len(take_s))]
                sel = np.concatenate([take_s, take_p])
                counts["mlp"] = int(len(take_s)); counts["mlp_pair"] = int(len(take_p))
                for i in sel:
                    picks.append((bank, int(i), recs[i]["family"]))
            else:
                rows = np.array(by.get(fam, []), dtype=np.int64)
                assert len(rows) >= n_per_family, f"{fam}: only {len(rows)} rows in {bank} < {n_per_family}"
                sel = np.sort(rng.permutation(rows)[:n_per_family])
                counts[fam] = int(len(sel))
                for i in sel:
                    picks.append((bank, int(i), fam))
        print(f"[mix_eq] {bank}: {n} rows; families {{k: len(v) for k, v in by.items()}} -> picked {sum(1 for p in picks if p[0] == bank)} ({time.time() - t0:.0f}s)", flush=True)
        del recs
    N = len(picks)
    order = rng.permutation(N)
    picks = [picks[i] for i in order]
    # vectors: gather per source in sorted-row chunks, scatter to the shuffled output positions
    vecs = np.memmap(f"{out}/vecs.f32.tmp", np.float32, "w+", shape=(N, D))
    for bank in (SRC_5, SRC_MLP):
        n_src = os.path.getsize(f"{bank}/vecs.f32") // (4 * D)
        src = np.memmap(f"{bank}/vecs.f32", np.float32, "r", shape=(n_src, D))
        idx = np.array([(k, p[1]) for k, p in enumerate(picks) if p[0] == bank], dtype=np.int64)
        idx = idx[np.argsort(idx[:, 1])]
        for c0 in range(0, len(idx), 50_000):
            ch = idx[c0:c0 + 50_000]
            vecs[ch[:, 0]] = src[ch[:, 1]]
        print(f"[mix_eq] vectors from {bank}: {len(idx)} rows ({time.time() - t0:.0f}s)", flush=True)
    vecs.flush()
    norms = np.linalg.norm(np.asarray(vecs[rng.choice(N, 4096, replace=False)]), axis=1)
    assert 0.98 < norms.min() and norms.max() < 1.02, f"unit-norm check failed: {norms.min()} {norms.max()}"
    # leak guard vs eval-cache v2 directions
    es = torch.load(EVAL_CACHE_V2, map_location="cpu", weights_only=False)
    dirs = [v for k, v in es.items() if isinstance(v, dict) and "dirs" in v and torch.is_tensor(v["dirs"])]
    if not dirs and isinstance(es.get("dirs"), dict):
        dirs = [v for v in es["dirs"].values() if torch.is_tensor(v)]
    if dirs:
        E = torch.nn.functional.normalize(torch.cat([d.float().reshape(-1, D) for d in dirs]), dim=1)
        mx = 0.0
        for c0 in range(0, N, 65_536):
            X = torch.from_numpy(np.asarray(vecs[c0:c0 + 65_536]))
            mx = max(mx, float((X @ E.T).max()))
        print(f"[mix_eq] leak guard: max cos vs {E.shape[0]} eval-cache dirs = {mx:.4f}", flush=True)
        assert mx < 0.999, f"leak: max cos {mx}"
    else:
        mx = None; print("[mix_eq] leak guard: eval-cache layout not recognised, skipped", flush=True)
    del vecs
    os.replace(f"{out}/vecs.f32.tmp", f"{out}/vecs.f32")
    # records: re-read sources lazily by line index
    src_lines = {}
    for bank in (SRC_5, SRC_MLP):
        with open(f"{bank}/records.jsonl") as f:
            src_lines[bank] = f.read().splitlines()
    with open(f"{out}/records.jsonl.tmp", "w") as f:
        for k, (bank, i, fam) in enumerate(picks):
            r = json.loads(src_lines[bank][i]); r["src_bank"] = bank; r["src_vec_idx"] = r.get("vec_idx", i); r["vec_idx"] = k
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    fam_counts = {}
    for _, _, fam in picks:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    stats = {"kind": "equal-share 6-family SFT midtrain mix (mlp singles first, pairs fill): everything_eq240k + mlp42_big", "n_examples": N, "n_vecs": N,
             "families": fam_counts, "n_per_family": n_per_family, "seed": seed, "sources": {SRC_5: "realact/realact_long/sae/bsf/cluster", SRC_MLP: "mlp/mlp_pair"},
             "layout": "seeded shuffle of all rows (records.jsonl line i == vec_idx i; src_bank/src_vec_idx point back)", "leak_guard_max_cos": mx,
             "d_model": D, "model": "Qwen/Qwen3.6-27B", "layer": 42, "created": time.time(), "wall_s": time.time() - t0}
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    json.dump({**stats, "trainer_args": {"--data-dir": out}}, open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    print(f"[mix_eq] DONE {out}: {N} rows {fam_counts} in {time.time() - t0:.0f}s", flush=True)
    return stats
