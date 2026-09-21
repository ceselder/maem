"""SIMPLE2M-V2 MASTER: retrain the 2M SAE on the FIXED streamer (1B distinct tokens) and rebuild everything downstream, then hand off to
simple2m_v2_chain_driver.py (compose -> SFT -> compose -> RL). Idempotent: every stage's call id/result lives in
~/shared/overnight/simple2m_v2/ids.json; rerun = resume. Discord at every stage boundary.

    cd /tmp && MODAL_PROFILE=safety-sahan nohup ~/modal_venv/bin/python3 ~/maemm-pub-simple2m/scripts/launchers/simple2m_v2_master.py >> ~/shared/overnight/simple2m_v2/master.out 2>&1 &

Stages: sae_train (maemm-sae2m-v3 train_online, 8xB200, ~9 h) -> sae_merge (CPU) -> sae_verify (1xB200) || sae_maxacts (8xB200, ~5 h)
        -> eval_cache_v4 (maemm-eval-cache-v4-s2m) -> bank_sft || bank_rl (maemm-sae2m-bank-s2m, 8xB200 each) -> chain driver.
"""
import json
import os
import subprocess
import sys
import time

import modal

IDS = os.path.expanduser("~/shared/overnight/simple2m_v2/ids.json")
LOG = os.path.expanduser("~/shared/overnight/simple2m_v2/master.log")
ROOT = "/data/sae2m_v2"
SHARDS, TRAINER0 = f"{ROOT}/shards", f"{ROOT}/trainer_0"
AE, MAXACTS, SPLIT = f"{TRAINER0}/ae.pt", f"{ROOT}/maxacts_top5.pt", f"{ROOT}/feature_split.npz"
V4 = "/data/eval_universal_ho/eval_sets_heldout_v4.pt"
APPS = {"sae": "maemm-sae2m-v3", "eval_cache": "maemm-eval-cache-v4-s2m", "bank": "maemm-sae2m-bank-s2m"}
TRAIN_KW = dict(total_tokens=1_000_000_000, save_every=60_000, keep_ckpts=1, resume=False, run_name="BatchTopK-2M-l42-online-v2", save_dir=SHARDS,
                max_hours=23.0, batch=4096, pool_rows=262_144, pool_reuse_max=1, micro_batch=16, dataset_skip=100_000, log_every=50, use_wandb=True)
BANK_KW = dict(ae=AE, maxacts=MAXACTS, windows_per_feature=3, min_tok=8, max_rows_per_family=2_330_000, seed=2026, feature_split=SPLIT)


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def log(m):
    line = f"[{now()}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def discord(m): subprocess.run(["notify-discord", f"simple2m-v2: {m}"])
def load(): return json.load(open(IDS))
def save(d): json.dump(d, open(IDS, "w"), indent=1)
def vol_has(path):
    """`path` may be the container path (/data/...) or the volume-root path; `modal volume ls` wants the latter."""
    vpath = path[len("/data"):] if path.startswith("/data/") else path
    r = subprocess.run(["modal", "volume", "ls", "maemm-data", os.path.dirname(vpath)], capture_output=True, text=True)
    return os.path.basename(vpath) in r.stdout


def wait_call(fc_id, what, poll=120, allow_fail=False):
    fc = modal.FunctionCall.from_id(fc_id); t0 = time.time()
    while True:
        try:
            r = fc.get(timeout=0); log(f"{what} DONE after {(time.time() - t0) / 60:.0f} min: {str(r)[:400]}"); return r
        except TimeoutError:
            time.sleep(poll)
        except Exception as e:  # noqa
            log(f"{what} FAILED: {type(e).__name__}: {str(e)[:600]}")
            if allow_fail: return None
            discord(f"{what} FAILED ({type(e).__name__}) — master paused; see master.log"); raise SystemExit(1)


def stage(d, key): return key in d and d[key].get("done")


def spawn(d, key, app, fn, note="", **kw):
    if key not in d:
        fc = modal.Function.from_name(app, fn).spawn(**kw)
        d[key] = {"call": fc.object_id, "app": app, "fn": fn, "kwargs": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in kw.items()}, "note": note, "spawned": now()}; save(d)
        log(f"{key}: spawned {app}.{fn} {fc.object_id}")
    return d[key]["call"]


d = load(); log("master start; stages done: " + ", ".join(k for k in d if isinstance(d[k], dict) and d[k].get("done")))
assert vol_has(SPLIT), f"{SPLIT} missing (seed it first)"

# ---- 1. SAE training on the fixed streamer -------------------------------------------------------------------------------------
if not stage(d, "sae_train"):
    fc = spawn(d, "sae_train", APPS["sae"], "train_online", note="1B DISTINCT tokens (skip 100k then shard by rank)", **TRAIN_KW)
    if "sae_train" in d and not d["sae_train"].get("announced"):
        discord(f"SAE-2M v2 training launched on the fixed streamer ({APPS['sae']}, 8xB200, ~9 h): {fc}"); d["sae_train"]["announced"] = True; save(d)
    r = wait_call(fc, "SAE train", allow_fail=True)
    if not vol_has(f"{SHARDS}/TRAIN_DONE"):
        if r is None or not (isinstance(r, dict) and r.get("train_done")):
            log("train call ended without TRAIN_DONE -> spawning resume leg")
            fc2 = modal.Function.from_name(APPS["sae"], "train_online").spawn(**{**TRAIN_KW, "resume": True})
            d["sae_train"].setdefault("legs", []).append(d["sae_train"]["call"]); d["sae_train"]["call"] = fc2.object_id; save(d)
            wait_call(fc2.object_id, "SAE train (resume leg)")
    assert vol_has(f"{SHARDS}/TRAIN_DONE"), "no TRAIN_DONE after the train call(s)"
    d["sae_train"].update({"done": True, "result": str(r)[:800] if r else None}); save(d)
    discord("SAE-2M v2 TRAIN_DONE -> merge")

# ---- 2. merge shards -> trainer_0/ae.pt -----------------------------------------------------------------------------------------
if not stage(d, "sae_merge"):
    fc = spawn(d, "sae_merge", APPS["sae"], "merge", save_dir=SHARDS, out_dir=TRAINER0)
    r = wait_call(fc, "SAE merge"); d["sae_merge"].update({"done": True, "result": str(r)[:800]}); save(d)

# ---- 3. verify || maxacts ------------------------------------------------------------------------------------------------------
if not stage(d, "sae_verify") or not stage(d, "sae_maxacts"):
    fv = spawn(d, "sae_verify", APPS["sae"], "verify", ae=AE, eval_tokens=20_000_000, dataset_skip=0, out_json=f"{ROOT}/verify.json")
    fm = spawn(d, "sae_maxacts", APPS["sae"], "maxacts", ae=AE, max_tokens=1_000_000_000, topn=5, window=32, dataset_skip=100_000,
               out_dir=f"{ROOT}/maxacts_parts", final=MAXACTS)
    if not stage(d, "sae_verify"):
        rv = wait_call(fv, "SAE verify"); d["sae_verify"].update({"done": True, "result": str(rv)[:800]}); save(d)
        discord(f"SAE-2M v2 verify: {str(rv)[:300]}")
    if not stage(d, "sae_maxacts"):
        rm = wait_call(fm, "SAE maxacts"); d["sae_maxacts"].update({"done": True, "result": str(rm)[:800]}); save(d)
        discord("SAE-2M v2 max-acts done -> eval cache v4 + SAE banks")

# ---- 4. eval cache v4 ------------------------------------------------------------------------------------------------------------
if not stage(d, "eval_cache_v4"):
    fc = spawn(d, "eval_cache_v4", APPS["eval_cache"], "build", n=512, seed=2026, out=V4, overwrite=False)
    r = wait_call(fc, "eval cache v4"); d["eval_cache_v4"].update({"done": True, "out": V4, "result": str(r)[:800]}); save(d)

# ---- 5. SAE banks (sft split || rl split) ------------------------------------------------------------------------------------------
if not stage(d, "bank_sae2m_sft") or not stage(d, "bank_sae2m_rl"):
    fs = spawn(d, "bank_sae2m_sft", APPS["bank"], "build", out_name="sae2m_v2_sft", split_key="sft", **BANK_KW)
    fr = spawn(d, "bank_sae2m_rl", APPS["bank"], "build", out_name="sae2m_v2_rl", split_key="rl", **BANK_KW)
    for key, fc in (("bank_sae2m_sft", fs), ("bank_sae2m_rl", fr)):
        if not stage(d, key):
            r = wait_call(fc, key); d[key].update({"done": True, "out": f"/data/banks/{d[key]['kwargs']['out_name']}", "result": str(r)[:800]}); save(d)
    discord("SAE-2M v2 banks built -> compose + SFT (chain driver)")

# ---- 6. hand off to the chain driver (compose_sft -> sft -> compose_rl -> rl) --------------------------------------------------------
if not d.get("chain_driver_launched"):
    subprocess.Popen(["nohup", os.path.expanduser("~/modal_venv/bin/python3"), os.path.expanduser("~/maemm-pub-simple2m/scripts/launchers/simple2m_v2_chain_driver.py")],
                     stdout=open(os.path.expanduser("~/shared/overnight/simple2m_v2/driver.out"), "a"), stderr=subprocess.STDOUT, cwd="/tmp", env={**os.environ, "MODAL_PROFILE": "safety-sahan"})
    d["chain_driver_launched"] = now(); save(d); log("chain driver launched")
log("master done")
