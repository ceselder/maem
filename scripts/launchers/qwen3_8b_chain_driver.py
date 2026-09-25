"""Qwen3-8B port of the simple2m chain (the 27B paper checkpoint's recipe, scaled down; branch qwen3-8b, model profile qwen3-8b).
Resumable: state in ~/shared/overnight/qwen3_8b/ids.json, log ~/shared/overnight/qwen3_8b/driver.log.

    A  layer-27 mean (maemm-8b-prep.whiten_mu, spawned by hand) -> 4 collections (maemm-8b-collect):
         SFT acts   1.2M rows  standalone 8-64-token windows, target = full context      Ultra-FineWeb docs [5.5M, ...)
         RL acts    300k rows  full-document forwards, ctx 64-2048                       docs [9.5M, ...)
         eval 8-64  32,768     same recipe as the SFT acts                               eval head, docs [0, ...)
         eval long  32,768     same recipe as the RL acts                                eval head, docs [50k, ...)
    B  eval cache /data/eval_universal_ho/eval_sets_8b.pt (maemm-8b-eval-cache) + the untrained base model as a step_0 checkpoint, scored
    C  SFT mix  = SAE bank sae8b_sft (Karvonen SFT-split features: enc + dec rows) + an EQUAL number of SFT acts (exact 50/50)
    D  full-FT SFT from Qwen3-8B (micro 32 x accum 16 x 8 = 4096/step, lr 1e-5, max_seq 192, 1 epoch, 8 intermediate ckpts) + evaluator
    E  RL pool  = SAE bank sae8b_rl (RL-split features) + an EQUAL number of RL acts
    F  full-parameter RL from the SFT final = the paper run's flags (CISPO, 2048 x 8, last-16-token reward, lr 5e-7, 300 steps, legacy
       uncentered reward via the app's MAEMM_SCORER_CENTER=0) minus the Qwen3.6-only --prefix-cache + evaluator

    cd /tmp && nohup python3 ~/maemm-pub-8b/scripts/launchers/qwen3_8b_chain_driver.py >> ~/shared/overnight/qwen3_8b/driver.out 2>&1 &
"""
import json
import math
import os
import subprocess
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
D_DIR = os.path.expanduser("~/shared/overnight/qwen3_8b")
IDS, LOG = f"{D_DIR}/ids.json", f"{D_DIR}/driver.log"
VOL = modal.Volume.from_name("maemm-data")
UFW = lambda skip: json.dumps({"dataset": "openbmb/Ultra-FineWeb", "config": None, "split": "en", "shuffle": False, "skip": skip})
CACHE = "/data/eval_universal_ho/eval_sets_8b.pt"
SAE16K_PEAKS_CALL = "fc-01M3D8WDE98CNY0PDYHVFYB068"   # maemm-8b-prep.sae16k_peaks (unseen-SAE corpus peaks)
EVAL_EXTRA = f"--eval-cache {CACHE} --no-extra-evals --eval-sae /data/qwen3_8b/sae16k/ae.pt"   # always-scored "sae" family = UNSEEN 16k SAE
SFT_RUN, SFT_MIX, RL_POOL = "qwen3_8b_sft", "mix8b_sft", "mix8b_rl"
RL_RUN = "qwen3_8b_rl_last16_lr5e-7"
EFF_BATCH, RL_STEPS = 4096, 300
SFT_EXTRA = "--full-ft --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --fsdp-prefetch 2"
RL_EXTRA = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
            "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 2048 "
            "--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 "
            "--score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
            f"--save-steps 25,50,100,150,200,250,300 --eval-cache {CACHE} --eval-sae /data/qwen3_8b/sae16k/ae.pt "
            "--reward-window-last 16 --lr 5e-7")
COLLECT = {"coll_sft": ("collect", dict(n_examples=1_200_000, out_name="ufw8b_ctx8_64_sft_1200k", batch=64, per_window=8, p_lo=8, p_hi=64, w_lo=8,
                                         w_hi=64, max_wins=4, seed=61, hf_stream=UFW(5_500_000), w_full=True)),
           "coll_rl": ("collect_fullctx", dict(n_examples=300_000, out_name="ufw8b_ctx64_2048_rl_300k", ctx_lo=64, ctx_hi=2048, per_doc=8, batch=8,
                                                tail=64, seed=62, hf_stream=UFW(9_500_000), family="realact_ctx64_2048")),
           "coll_ev864": ("collect", dict(n_examples=32_768, out_name="ufw8b_ctx8_64_evalhead_32k", batch=64, per_window=8, p_lo=8, p_hi=64, w_lo=8,
                                           w_hi=64, max_wins=4, chunk_examples=8192, seed=71, hf_stream=UFW(0), w_full=True)),
           "coll_ev2048": ("collect_fullctx", dict(n_examples=32_768, out_name="ufw8b_ctx64_2048_evalhead_32k", ctx_lo=64, ctx_hi=2048, per_doc=8,
                                                    batch=8, tail=64, chunk_examples=8192, seed=72, hf_stream=UFW(50_000), family="realact_ctx64_2048"))}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{now()} {m}\n")
    print(m, flush=True)


def discord(m):
    try:
        subprocess.run(["notify-discord", f"[qwen3-8b chain] {m}"], timeout=30)
    except Exception:
        pass


def load():
    return json.load(open(IDS)) if os.path.exists(IDS) else {}


def save(d):
    d["updated"] = now(); json.dump(d, open(IDS + ".tmp", "w"), indent=1); os.replace(IDS + ".tmp", IDS)


def vol_json(path):
    return json.loads(b"".join(VOL.read_file(path.replace("/data/", "", 1))))


def vol_has(path):
    try:
        vol_json(path) if path.endswith(".json") else list(VOL.listdir(path.replace("/data/", "", 1)))
        return True
    except Exception:
        return False


def wait_call(cid, what):
    t0 = time.time()
    while True:
        try:
            r = modal.FunctionCall.from_id(cid).get(timeout=60)
            log(f"{what} done after {time.time() - t0:.0f}s: {str(r)[:300]}"); return r
        except TimeoutError:
            continue
        except Exception as e:
            log(f"{what} FAILED: {type(e).__name__}: {str(e)[:600]}"); discord(f"{what} FAILED: {type(e).__name__}"); raise


def wait_bank(d, key):
    """collector calls can raise at teardown after every rank finished (27B lesson): trust build_stats.json, else finalize_only."""
    out = f"/data/banks/{COLLECT[key][1]['out_name']}"
    try:
        wait_call(d[key]["call"], key)
    except Exception:
        if not vol_has(f"{out}/build_stats.json"):
            log(f"{key}: no build_stats -> finalize_only")
            a = COLLECT[key][1]
            c = modal.Function.from_name("maemm-8b-collect", "finalize_only").spawn(out_name=a["out_name"], n_examples=a["n_examples"], seed=a["seed"])
            wait_call(c.object_id, f"{key} finalize_only")
    st = vol_json(f"{out}/build_stats.json"); d[key]["n_examples"] = st["n_examples"]; save(d)
    return st


def main():
    d = load()
    # ---- A. mean -> collections ----
    if "collections_spawned" not in d:
        wait_call(d["whiten_mu_call"], "whiten_mu")
        for key, (fn, args) in COLLECT.items():
            if key not in d:
                c = modal.Function.from_name("maemm-8b-collect", fn).spawn(**args)
                d[key] = {"call": c.object_id, "fn": fn, "args": args, "spawned": now()}; save(d); log(f"{key}: {c.object_id}")
        d["collections_spawned"] = now(); save(d)
    # ---- B. eval cache + untrained base ----
    if "eval_cache" not in d:
        for key in ("coll_ev864", "coll_ev2048"):
            wait_bank(d, key)
        wait_call(SAE16K_PEAKS_CALL, "16k-SAE corpus peaks")
        info = modal.Function.from_name("maemm-8b-eval-cache", "build").remote()
        d["eval_cache"] = {"path": CACHE, "info": info, "built": now()}; save(d); log(f"eval cache built: {json.dumps(info)[:400]}")
    if "base_eval" not in d:
        mk = modal.Function.from_name("maemm-eval-cache-ctx864-s2m", "make_base_ckpt").remote(out="/data/base_ckpt/qwen3_8b/step_0", model="Qwen/Qwen3-8B")
        c = modal.Function.from_name("maemm-8b-eval", "daemon").spawn(ckpt_dir="/data/base_ckpt/qwen3_8b", tag="base_qwen3_8b", once=True, only_step=0,
                                                                     final_step=0, extra_args=f"--full-model --no-wandb --dump-per-dir {EVAL_EXTRA}")
        d["base_eval"] = {"ckpt": mk, "call": c.object_id, "spawned": now()}; save(d); log(f"base eval: {c.object_id}")
    # ---- C/D. SFT mix + SFT ----
    if "sft" not in d:
        if "compose_sft" not in d:
            wait_call(d["sae_banks"]["sae8b_sft"]["call"], "SAE bank sae8b_sft")
            fam = vol_json("/data/banks/sae8b_sft/build_stats.json")["families"]
            n_sae = int(fam["sae8b"]) + int(fam["sae8b_dec"])
            acts = wait_bank(d, "coll_sft")["n_examples"]
            n_acts = min(n_sae, int(acts)) // 2 * 2
            spec = [{"name": "acts8b_sft", "banks": ["/data/banks/ufw8b_ctx8_64_sft_1200k"], "dtype": "f16", "families": ["realact"], "n": n_acts,
                     "all_one_family": "realact", "fresh": "Ultra-FineWeb docs [5.5M, ...); standalone 64-token windows, ctx 8-64, target = FULL context"},
                    {"name": "sae8b_sft", "banks": ["/data/banks/sae8b_sft"], "dtype": "f32", "families": ["sae8b", "sae8b_dec"],
                     "n": {"sae8b": n_acts // 2, "sae8b_dec": n_acts // 2},
                     "fresh": "Karvonen L27 SAE, SFT split (55,687 features; eval 3,276 + RL 6,552 held out); enc + dec x end-anchored max-act windows"}]
            c = modal.Function.from_name("maemm-8b-mix", "build").spawn(out_name=SFT_MIX, spec_json=json.dumps(spec), seed=2040, threads=48,
                                                                        dense_frac=0.0, eval_cache=CACHE)
            d["compose_sft"] = {"call": c.object_id, "spec": spec, "n_acts": n_acts, "n_sae_avail": n_sae, "spawned": now()}; save(d)
            log(f"compose SFT {c.object_id}: acts {n_acts} + sae {n_acts} (sae avail {n_sae})")
        wait_call(d["compose_sft"]["call"], "compose SFT")
        st = vol_json(f"/data/banks/{SFT_MIX}/build_stats.json"); n_rows = int(st["n_examples"]); steps = math.ceil(n_rows / EFF_BATCH)
        t = modal.Function.from_name("maemm-8b-sft", "train").spawn(run_name=SFT_RUN, data_dir=f"/data/banks/{SFT_MIX}", n_ckpts=8, epochs=1,
                                                                   batch_size=32, lr=1e-5, max_seq=192, backend="nccl", extra_args=SFT_EXTRA)
        e = modal.Function.from_name("maemm-8b-eval", "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{SFT_RUN}", tag=f"sft_{SFT_RUN}_c",
                                                                               wandb_name=f"{SFT_RUN}_eval", final_step=steps, extra_args=EVAL_EXTRA)
        d["sft"] = {"train": t.object_id, "eval": e.object_id, "rows": n_rows, "families": st["families"], "steps_expected": steps,
                    "save": f"/data/sft_mix/{SFT_RUN}", "extra": SFT_EXTRA, "spawned": now()}; save(d)
        log(f"SFT spawned: train {t.object_id} eval {e.object_id}: {n_rows} rows, {steps} steps")
    # ---- E. RL pool ----
    if "compose_rl" not in d:
        wait_call(d["sae_banks"]["sae8b_rl"]["call"], "SAE bank sae8b_rl")
        fam = vol_json("/data/banks/sae8b_rl/build_stats.json")["families"]
        n_sae = int(fam["sae8b"]) + int(fam["sae8b_dec"])
        acts = wait_bank(d, "coll_rl")["n_examples"]
        n_each = min(n_sae, int(acts)) // 2 * 2
        spec = [{"name": "acts8b_rl", "banks": ["/data/banks/ufw8b_ctx64_2048_rl_300k"], "dtype": "f16", "families": ["realact_ctx64_2048"], "n": n_each,
                 "all_one_family": "realact_ctx64_2048", "fresh": "Ultra-FineWeb docs [9.5M, ...); FULL-document forwards, ctx 64-2048"},
                {"name": "sae8b_rl", "banks": ["/data/banks/sae8b_rl"], "dtype": "f32", "families": ["sae8b", "sae8b_dec"],
                 "n": {"sae8b": n_each // 2, "sae8b_dec": n_each // 2}, "fresh": "Karvonen L27 SAE, RL split (6,552 features; disjoint from SFT + eval)"}]
        c = modal.Function.from_name("maemm-8b-mix", "build").spawn(out_name=RL_POOL, spec_json=json.dumps(spec), seed=2041, threads=48,
                                                                    dense_frac=0.0, eval_cache=CACHE)
        d["compose_rl"] = {"call": c.object_id, "spec": spec, "n_each": n_each, "spawned": now()}; save(d)
        log(f"compose RL {c.object_id}: acts {n_each} + sae {n_each}")
        wait_call(c.object_id, "compose RL")
    # ---- F. RL ----
    if "rl" not in d:
        wait_call(d["sft"]["train"], "SFT train")
        final = f"/data/sft_mix/{SFT_RUN}/final"
        for _ in range(40):
            if vol_has(f"{final}/config.json"):
                break
            time.sleep(60)
        save_dir = f"/data/ckpts_{RL_RUN}"
        extra = f"{RL_EXTRA} --run-name {RL_RUN} --save-dir {save_dir}"
        t = modal.Function.from_name("maemm-8b-rl", "train").spawn(n_rollout=2, n_trainer=6, total_steps=RL_STEPS, extra_args=extra,
                                                                  pool_dir=f"/data/banks/{RL_POOL}", policy_base=final, full_param=True)
        e = modal.Function.from_name("maemm-8b-eval", "fullmodel_daemon").spawn(ckpt_dir=save_dir, tag=f"{RL_RUN}_c", wandb_name=f"{RL_RUN}_eval",
                                                                               final_step=RL_STEPS, extra_args=EVAL_EXTRA)
        d["rl"] = {"train": t.object_id, "eval": e.object_id, "save": save_dir, "policy_base": final, "extra": extra, "spawned": now()}; save(d)
        log(f"RL spawned: train {t.object_id} eval {e.object_id}")
        discord(f"Qwen3-8B SFT done; RL launched ({RL_RUN}, 300 steps)")
    wait_call(d["rl"]["train"], "RL train")
    discord("Qwen3-8B RL finished; evaluator finishing the last checkpoints")


if __name__ == "__main__":
    main()
