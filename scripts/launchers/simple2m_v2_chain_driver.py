"""SIMPLE2M-V2 chain driver (2026-09-21): the SAME recipe as simple2m_chain_driver.py, re-run end-to-end on the 2M SAE RETRAINED with the
fixed data streamer (1B DISTINCT tokens; /data/sae2m_v2). Inputs (all produced by simple2m_v2_master.py before this starts): SAE banks
/data/banks/sae2m_v2_sft + sae2m_v2_rl, eval cache v4 (= v2 families + held-out slice families of the NEW dictionary), and the two UNCHANGED
activation banks (ufw_ctx8_64_sft_4m_v3, ufw_ctx64_2048_rl_500k_v3). RL = LEGACY reward (same as the 8M main arm); evals centered (-c app).

--- original header follows ---
SIMPLE2M chain driver (paper-clean recipe, 2026-09-17): NO pretrain/midtrain distinction.

  SFT  = ONE full-FT run from the BASE model on an exact 50/50 mix: N Ultra-FineWeb activations of 8-64 tokens of context whose
         TARGET IS THE FULL CONTEXT (standalone window, w_full: re-encoding the target reproduces the exact input -> precise inversion)
         + N 2M-SAE rows (N/2 encoder-column + N/2 decoder-row directions, max-act windows of the SFT feature split), N = min(4M, avail)
  RL   = full-parameter CISPO/GRPO (8 x 2048, lr 1e-6, 300 steps, reward = max cosine over the WHOLE rollout span) on an EVEN
         split (paper sec. RL): 1/2 FULL-DOCUMENT-forward activations harvested INDEPENDENT of context length (positions uniform over
         the first 2048 tokens of each document, contexts 64-2048) and 1/2 SAE rows (RL feature split); every row from documents /
         features disjoint from SFT and eval; every bank publishes doc_ids_used.json (Ultra-FineWeb document registry)
  eval = cache v3 (v2 families unchanged for mean_all + held-out 2M-SAE enc/dec slice families), evaluator maemm-eval-ckpt-s2m

Inputs (spawned by hand, ids in ~/shared/overnight/simple2m/ids.json): bank_sae2m_sft, bank_sae2m_rl (maemm-sae2m-bank-s2m),
coll_sft_ctx8_64 (collect, w_full), coll_rl_ctx64_2048 (collect_fullctx, one unstratified bank) on maemm-collect-bank-s2m. This driver
waits for them and runs: compose SFT mix -> SFT + evaluator | compose RL pool | (SFT final) -> RL + evaluator -> done.
Idempotent: every stage is recorded in ids.json; rerun = resume. Discord on launches / completion / failure.

    cd /tmp && MODAL_PROFILE=safety-sahan nohup ~/modal_venv/bin/python3 ~/maemm-pub-simple2m/scripts/launchers/simple2m_v2_chain_driver.py >> ~/shared/overnight/simple2m_v2/driver.out 2>&1 &
"""
import json
import math
import os
import subprocess
import sys
import time

import modal

IDS = os.path.expanduser("~/shared/overnight/simple2m_v2/ids.json")
LOG = os.path.expanduser("~/shared/overnight/simple2m_v2/driver.log")
V3 = "/data/eval_universal_ho/eval_sets_heldout_v4.pt"   # v4 = v2 families + slice families of the v2 SAE (variable kept as V3 for the shared code below)
APPS = {"bank": "maemm-sae2m-bank-s2m", "collect": "maemm-collect-bank-s2m", "store": "maemm-acts-ufw-s2m", "long": "maemm-ufw-long-bank-s2m",
        "compose": "maemm-mix-5m-bank-s2m", "sft": "maemm-sft-fullft-s2m", "eval": "maemm-eval-ckpt-s2m-c", "rl": "maemm-rl-disagg-fullparam-s2m"}   # eval -c = centered scorer; rl = legacy reward
SFT_RUN, SFT_MIX = "simple2m_v2_sft", "mix_simple2m_v2_sft"
RL_RUN, RL_POOL = "rl_simple2m_v2_8x2048_anywin", "mix_simple2m_v2_rl"
EFF_BATCH = 4096                                   # micro 32 x grad-accum 16 x 8 GPUs
SFT_EXTRA = ("--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --prefix-share-step "
             "--fsdp-prefetch 2")                  # NO --init-adapter: the policy starts from the base model
RL_RECIPE = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
             "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 2048 "
             "--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 0 "
             "--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
             "--lr 1e-6 --save-steps 25,50,100,150,200,250,300 --eval-cache " + V3)   # inline eval on cache v3 too
RL_STEPS = 300


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(m):
    line = f"[{now()}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def discord(m):
    subprocess.run(["notify-discord", f"simple2m-v2: {m}"])


def load():
    return json.load(open(IDS))


def save(d):
    json.dump(d, open(IDS, "w"), indent=1)


def wait_call(fc_id, what, poll=120):
    fc = modal.FunctionCall.from_id(fc_id)
    t0 = time.time()
    while True:
        try:
            r = fc.get(timeout=0)
            log(f"{what} DONE after {(time.time() - t0) / 60:.0f} min: {str(r)[:400]}")
            return r
        except TimeoutError:
            time.sleep(poll)
        except Exception as e:  # noqa
            log(f"{what} FAILED: {type(e).__name__}: {str(e)[:600]}")
            discord(f"{what} FAILED ({type(e).__name__}) — chain paused; see driver.log")
            raise SystemExit(1)


def vol_get_json(path):
    out = f"/tmp/simple2m_{path.strip('/').replace('/', '_')}"
    subprocess.run(["modal", "volume", "get", "--force", "maemm-data", path, out], check=True, capture_output=True)
    return json.load(open(out))


def vol_has(path):
    r = subprocess.run(["modal", "volume", "ls", "maemm-data", os.path.dirname(path)], capture_output=True, text=True)
    return os.path.basename(path) in r.stdout


def stage(d, key):
    return key in d and d[key].get("done")


def _manifests_done(out_name, world=8):
    ok = 0
    for r in range(world):
        try:
            m = vol_get_json(f"/banks/{out_name}/shards/manifest_r{r}.json")
            ok += int(bool(m.get("done")) and int(m.get("kept", 0)) >= int(m.get("n_examples_target", 1)))
        except Exception:  # noqa
            pass
    return ok == world


def wait_collect(d, key, out_name, n_examples, seed, poll=120):
    """Wait for a collection call. If the call FAILED but every rank left a DONE manifest (the interpreter-teardown crash that hit
    the SFT collection at 03:24Z), publish the bank with maemm-collect-bank-s2m.finalize_only and wait for that instead."""
    fc = modal.FunctionCall.from_id(d[key]["call"]); t0 = time.time()
    while True:
        try:
            r = fc.get(timeout=0)
            log(f"{key} DONE after {(time.time() - t0) / 60:.0f} min: {str(r)[:300]}"); return r
        except TimeoutError:
            time.sleep(poll)
        except Exception as e:  # noqa
            log(f"{key} call FAILED: {type(e).__name__}: {str(e)[:300]}")
            if d[key].get("finalize_spawned") or not _manifests_done(out_name):
                discord(f"{key} FAILED and its shards are incomplete — chain paused"); raise SystemExit(1)
            c = modal.Function.from_name(APPS["collect"], "finalize_only").spawn(out_name=out_name, n_examples=n_examples, seed=seed)
            d[key]["collect_call"] = d[key]["call"]; d[key]["call"] = c.object_id; d[key]["finalize_spawned"] = now(); save(d)
            log(f"{key}: all ranks have DONE manifests -> finalize_only spawned {c.object_id}")
            fc = modal.FunctionCall.from_id(c.object_id)


# ---------------------------------------------------------------------------------------------------------------------
d = load()
log("driver start; stages done: " + ", ".join(k for k in d if isinstance(d[k], dict) and d[k].get("done")))

# ---- B. SFT mix (exact 50/50) -> SFT + evaluator ----------------------------------------------------------------------
if not stage(d, "compose_sft"):
    for need in ("/banks/sae2m_v2_sft/build_stats.json", "/banks/ufw_ctx8_64_sft_4m_v3/build_stats.json"):
        assert vol_has(need), f"input bank missing: {need}"
    st = vol_get_json("/banks/sae2m_v2_sft/build_stats.json"); fam = st["families"]
    n_enc, n_dec = int(fam["sae2m"]), int(fam["sae2m_dec"])
    acts_avail = int(vol_get_json("/banks/ufw_ctx8_64_sft_4m_v3/build_stats.json")["n_examples"])
    n_sae = min(n_enc + n_dec, acts_avail) // 2 * 2                    # even, so enc == dec
    n_acts = n_sae                                                       # EXACT 50/50: activations == SAE rows
    spec = [{"name": "acts_ufw_sft", "banks": ["/data/banks/ufw_ctx8_64_sft_4m_v3"], "dtype": "f16", "families": ["realact"], "n": n_acts,
             "all_one_family": "realact", "fresh": "Ultra-FineWeb docs [5.5M, ...) ordered stream; standalone 64-token windows, ctx 8-64, target = FULL context"},
            {"name": "sae2m_sft", "banks": ["/data/banks/sae2m_v2_sft"], "dtype": "f32", "families": ["sae2m", "sae2m_dec"],
             "n": {"sae2m": n_sae // 2, "sae2m_dec": n_sae // 2},
             "fresh": "v2 2M-SAE (1B distinct tokens) SFT feature split (1,847,152 features; eval 100k + RL 150k held out); enc + dec directions x end-anchored max-act windows"}]
    if "compose_sft" not in d:
        c = modal.Function.from_name(APPS["compose"], "build").spawn(out_name=SFT_MIX, spec_json=json.dumps(spec), seed=2040, threads=48,
                                                                      dense_frac=0.0, eval_cache=V3)
        d["compose_sft"] = {"call": c.object_id, "out": f"/data/banks/{SFT_MIX}", "spec": spec, "n_acts": n_acts, "n_sae": n_sae,
                            "sae_bank_rows": {"sae2m": n_enc, "sae2m_dec": n_dec}, "acts_avail": acts_avail, "spawned": now()}; save(d)
        log(f"compose SFT spawned {c.object_id}: acts {n_acts} + sae {n_sae} (bank had enc {n_enc} dec {n_dec}; acts avail {acts_avail})")
    wait_call(d["compose_sft"]["call"], f"compose {SFT_MIX}")
    st = vol_get_json(f"/banks/{SFT_MIX}/build_stats.json")
    d["compose_sft"].update({"done": True, "n_examples": st["n_examples"], "families": st["families"]}); save(d)
    discord(f"SFT mix composed: {st['n_examples']} rows {st['families']} -> launching full-FT SFT from base")

if "sft" not in d:
    n_rows = d["compose_sft"]["n_examples"]; steps = math.ceil(n_rows / EFF_BATCH)
    if "sft" not in d:
        t = modal.Function.from_name(APPS["sft"], "train").spawn(run_name=SFT_RUN, data_dir=f"/data/banks/{SFT_MIX}", n_ckpts=8, epochs=1,
                                                                  batch_size=32, lr=1e-5, max_seq=192, backend="nccl", extra_args=SFT_EXTRA)
        e = modal.Function.from_name(APPS["eval"], "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{SFT_RUN}", tag=f"sft_{SFT_RUN}",
                                                                              wandb_name=f"{SFT_RUN}_eval", final_step=steps,
                                                                              extra_args=f"--eval-cache {V3} --no-extra-evals")
        d["sft"] = {"train": t.object_id, "eval": e.object_id, "run": SFT_RUN, "save": f"/data/sft_mix/{SFT_RUN}", "data": f"/data/banks/{SFT_MIX}",
                    "eff_batch": EFF_BATCH, "micro_batch": 32, "grad_accum": 16, "lr": 1e-5, "max_seq": 192, "steps_expected": steps, "init": "base model",
                    "extra": SFT_EXTRA, "app": APPS["sft"], "eval_app": APPS["eval"], "eval_cache": V3, "spawned": now()}; save(d)
        log(f"SFT spawned: train {t.object_id} eval {e.object_id} ({steps} steps expected)")
        discord(f"SFT launched from BASE on {SFT_MIX} ({n_rows} rows 50/50, {steps} steps ≈ {steps * 12 / 3600:.1f} h at 12 s/step); RL follows")

# ---- C. RL pool (exact thirds) -------------------------------------------------------------------------------------------
if not stage(d, "compose_rl"):
    for need in ("/banks/sae2m_v2_rl/build_stats.json", "/banks/ufw_ctx64_2048_rl_500k_v3/build_stats.json"):
        assert vol_has(need), f"input bank missing: {need}"
    fam = vol_get_json("/banks/sae2m_v2_rl/build_stats.json")["families"]
    n_sae = int(fam["sae2m"]) + int(fam["sae2m_dec"])
    n_avail = int(vol_get_json("/banks/ufw_ctx64_2048_rl_500k_v3/build_stats.json")["n_examples"])
    n_each = min(n_sae, n_avail)                                         # EVEN split: activations == SAE rows
    spec = [{"name": "acts_ctx64_2048_rl", "banks": ["/data/banks/ufw_ctx64_2048_rl_500k_v3"], "dtype": "f16", "families": ["realact_ctx64_2048"], "n": n_each,
             "all_one_family": "realact_ctx64_2048",
             "fresh": "Ultra-FineWeb docs [9.5M, ...) ordered stream; FULL-document forwards (first 2048 tokens), positions uniform, ctx 64-2048"},
            {"name": "sae2m_rl", "banks": ["/data/banks/sae2m_v2_rl"], "dtype": "f32", "families": ["sae2m", "sae2m_dec"],
             "n": {"sae2m": n_each // 2, "sae2m_dec": n_each - n_each // 2} if n_each < n_sae else None,
             "fresh": "2M-SAE RL feature split (150,000 features; disjoint from SFT and eval); enc + dec"}]
    if "compose_rl" not in d:
        c = modal.Function.from_name(APPS["compose"], "build").spawn(out_name=RL_POOL, spec_json=json.dumps(spec), seed=2041, threads=48,
                                                                      dense_frac=0.0, eval_cache=V3)
        d["compose_rl"] = {"call": c.object_id, "out": f"/data/banks/{RL_POOL}", "spec": spec, "n_acts": n_each, "n_sae": n_each,
                           "avail": {"sae": n_sae, "acts": n_avail}, "spawned": now()}; save(d)
        log(f"compose RL spawned {c.object_id}: acts {n_each} (ctx 64-2048, unstratified) + sae {n_each} (avail sae {n_sae} acts {n_avail})")
    wait_call(d["compose_rl"]["call"], f"compose {RL_POOL}")
    st = vol_get_json(f"/banks/{RL_POOL}/build_stats.json")
    d["compose_rl"].update({"done": True, "n_examples": st["n_examples"], "families": st["families"]}); save(d)
    log(f"RL pool: {st['n_examples']} rows {st['families']}")

# ---- D. full-parameter RL (all-token reward) + evaluator ---------------------------------------------------------------
if not stage(d, "rl"):
    if not d["sft"].get("done"):
        try:
            wait_call(json.load(open(IDS))["sft"]["train"], "SFT train")
        except SystemExit:
            log("SFT call failed -> polling /sft_mix/<run>/final/SAVE_DONE instead (a resume leg may finish it)")
            t0 = time.time()
            while not vol_has(f"/sft_mix/{SFT_RUN}/final/SAVE_DONE"):
                if time.time() - t0 > 30 * 3600:
                    discord("SFT final never appeared (30 h) — chain paused"); raise SystemExit(1)
                time.sleep(300)
        ok = False
        for _ in range(40):
            if vol_has(f"/sft_mix/{SFT_RUN}/final/SAVE_DONE"):
                ok = True; break
            time.sleep(60)
        if not ok:
            discord("SFT final has no SAVE_DONE — RL NOT launched"); log("SFT final missing"); raise SystemExit(1)
        d["sft"]["done"] = True; save(d)
    save_dir = f"/data/ckpts_{RL_RUN}"
    # RL init = the SFT final by default; ids.json["rl_policy_base"] (e.g. "/data/sft_mix/simple2m_sft/step_244") overrides it (early stopping
    # on the held-out eval: the SFT's held-out metrics declined monotonically after step 244 on 2026-09-17)
    policy_base = d.get("rl_policy_base") or f"/data/sft_mix/{SFT_RUN}/final"
    if "rl" not in d:
        extra = f"{RL_RECIPE} --run-name {RL_RUN} --save-dir {save_dir}"
        t = modal.Function.from_name(APPS["rl"], "train").spawn(n_rollout=2, n_trainer=6, total_steps=RL_STEPS, extra_args=extra,
                                                                 pool_dir=f"/data/banks/{RL_POOL}", policy_base=policy_base, full_param=True)
        e = modal.Function.from_name(APPS["eval"], "fullmodel_daemon").spawn(ckpt_dir=save_dir, tag=RL_RUN, wandb_name=f"{RL_RUN}_eval", final_step=RL_STEPS,
                                                                              extra_args=f"--eval-cache {V3} --no-extra-evals")
        d["rl"] = {"train": t.object_id, "eval": e.object_id, "run": RL_RUN, "save": save_dir, "policy_base": policy_base,
                   "pool": f"/data/banks/{RL_POOL}", "lr": "1e-6", "group_size": 8, "groups_per_step": 2048, "steps": RL_STEPS, "split": "2+6",
                   "reward": "LEGACY max cosine over the WHOLE rollout span (raw-activation cosine, = the 8M main arm; --reward-window-last 0); EVAL centered on cache v4", "extra": extra, "app": APPS["rl"], "eval_app": APPS["eval"],
                   "eval_cache": V3, "spawned": now()}; save(d)
        log(f"RL spawned: train {t.object_id} eval {e.object_id}")
        discord(f"SFT DONE -> full-param RL launched from {policy_base}: {RL_RUN} (lr 1e-6, 2+6, 8x2048, whole-span reward, 300 steps) on {RL_POOL}; train {t.object_id}")
    wait_call(d["rl"]["train"], "RL train")
    d["rl"]["done"] = True; save(d)
    discord(f"RL DONE: {RL_RUN} 300 steps; evaluator {d['rl']['eval']} scores the last checkpoints (cache v3: mean_all + held-out 2M enc/dec)")
log("chain complete")
