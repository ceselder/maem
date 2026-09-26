"""Qwen3.6-27B chain trained IN-DISTRIBUTION on the 131k layer-42 SAE (the v3 eval cache's "sae" family), mirroring the Qwen3-8B recipe
(~2M-row 50/50 SFT; 50/50 RL pool; the 27B paper checkpoint's SFT + RL flags incl. the prefix cache and the legacy RL reward).
Resumable: state ~/shared/overnight/sae131k_indist/ids.json, log driver.log. Only ever run ONE instance (check `ps aux | grep sae131k_chain`).

    C  SFT mix  = sae131k_sft bank (enc + dec rows, SFT-split features) + an EQUAL number of 8-64-token acts from the paper chain's SFT bank
    D  full-FT SFT from Qwen3.6-27B (paper flags) + evaluators: v3 cache (all families; 131k sae = in-distribution, 2M slices = unseen)
       and the ctx864 cache (held-out 8-64-token windows)
    E  RL pool  = sae131k_rl bank + an EQUAL number of 64-2048-token acts from the paper chain's RL bank
    F  full-parameter RL from the SFT final (paper flags: CISPO, 2048 x 8, last-16 reward, lr 5e-7, 300 steps, legacy reward) + evaluators

    cd /tmp && nohup python3 ~/maemm-pub-131k/scripts/launchers/sae131k_chain_driver.py >> ~/shared/overnight/sae131k_indist/driver.out 2>&1 &
"""
import json
import math
import os
import subprocess
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
D_DIR = os.path.expanduser("~/shared/overnight/sae131k_indist")
IDS, LOG = f"{D_DIR}/ids.json", f"{D_DIR}/driver.log"
VOL = modal.Volume.from_name("maemm-data")
V3 = "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
CTX864 = "/data/eval_universal_ho/eval_sets_ctx864.pt"
APPS = {"sft": "maemm-131k-sft", "rl": "maemm-131k-rl", "mix": "maemm-mix-5m-bank-s2m", "eval": "maemm-eval-ckpt-s2m-c"}
SFT_RUN, SFT_MIX, RL_POOL, RL_RUN = "sae131k_sft", "mix131k_sft", "mix131k_rl", "rl_sae131k_8x2048_last16_lr5e-7"
EFF_BATCH, RL_STEPS = 4096, 300
SFT_EXTRA = ("--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --prefix-share-step "
             "--fsdp-prefetch 2")
RL_EXTRA = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
            "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 2048 "
            "--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 "
            "--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
            f"--save-steps 25,50,100,150,200,250,300 --eval-cache {V3} --reward-window-last 16 --lr 5e-7")
EVAL_V3 = f"--eval-cache {V3} --no-extra-evals"
EVAL_CTX = f"--eval-cache {CTX864} --no-extra-evals --dump-per-dir"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{now()} {m}\n")
    print(m, flush=True)


def discord(m):
    try:
        subprocess.run(["notify-discord", f"[27B in-dist 131k chain] {m}"], timeout=30)
    except Exception:
        pass


def load():
    return json.load(open(IDS))


def save(d):
    d["updated"] = now(); json.dump(d, open(IDS + ".tmp", "w"), indent=1); os.replace(IDS + ".tmp", IDS)


def vol_json(path):
    return json.loads(b"".join(VOL.read_file(path.replace("/data/", "", 1))))


def vol_has(path):
    try:
        list(VOL.listdir(path.replace("/data/", "", 1))); return True
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


def evaluators(ckpt_dir, tag, final_step):
    ev = modal.Function.from_name(APPS["eval"], "fullmodel_daemon")
    return {"v3": ev.spawn(ckpt_dir=ckpt_dir, tag=f"{tag}_c", wandb_name=f"{tag}_eval", final_step=final_step, extra_args=EVAL_V3).object_id,
            "ctx864": ev.spawn(ckpt_dir=ckpt_dir, tag=f"{tag}_ctx864_c", wandb_name=f"{tag}_ctx864_eval", final_step=final_step,
                               extra_args=EVAL_CTX).object_id}


def main():
    d = load()
    if "sft" not in d:
        if "compose_sft" not in d:
            wait_call(d["sae_banks"]["sae131k_sft"]["call"], "SAE bank sae131k_sft")
            fam = vol_json("/data/banks/sae131k_sft/build_stats.json")["families"]
            n = (int(fam["sae131k"]) + int(fam["sae131k_dec"])) // 2 * 2
            spec = [{"name": "acts_ufw_sft", "banks": ["/data/banks/ufw_ctx8_64_sft_4m_v3"], "dtype": "f16", "families": ["realact"], "n": n,
                     "all_one_family": "realact", "fresh": "the paper chain's SFT acts: Ultra-FineWeb docs [5.5M, ...), standalone 64-token windows, ctx 8-64"},
                    {"name": "sae131k_sft", "banks": ["/data/banks/sae131k_sft"], "dtype": "f32", "families": ["sae131k", "sae131k_dec"],
                     "n": {"sae131k": n // 2, "sae131k_dec": n // 2},
                     "fresh": "131k L42 SAE, SFT split (104,701 features; eval 13,107 = all pool_heldout features + RL 13,107 held out); enc + dec"}]
            c = modal.Function.from_name(APPS["mix"], "build").spawn(out_name=SFT_MIX, spec_json=json.dumps(spec), seed=2040, threads=48,
                                                                     dense_frac=0.0, eval_cache=V3)
            d["compose_sft"] = {"call": c.object_id, "spec": spec, "n_acts": n, "spawned": now()}; save(d)
            log(f"compose SFT {c.object_id}: acts {n} + sae {n}")
        wait_call(d["compose_sft"]["call"], "compose SFT")
        st = vol_json(f"/data/banks/{SFT_MIX}/build_stats.json"); n_rows = int(st["n_examples"]); steps = math.ceil(n_rows / EFF_BATCH)
        t = modal.Function.from_name(APPS["sft"], "train").spawn(run_name=SFT_RUN, data_dir=f"/data/banks/{SFT_MIX}", n_ckpts=8, epochs=1,
                                                                 batch_size=32, lr=1e-5, max_seq=192, backend="nccl", extra_args=SFT_EXTRA)
        e = evaluators(f"/data/sft_mix/{SFT_RUN}", f"sft_{SFT_RUN}", steps)
        d["sft"] = {"train": t.object_id, "evals": e, "rows": n_rows, "families": st["families"], "steps_expected": steps,
                    "save": f"/data/sft_mix/{SFT_RUN}", "extra": SFT_EXTRA, "spawned": now()}; save(d)
        log(f"SFT spawned: train {t.object_id} evals {e}: {n_rows} rows, {steps} steps")
    if "compose_rl" not in d:
        wait_call(d["sae_banks"]["sae131k_rl"]["call"], "SAE bank sae131k_rl")
        fam = vol_json("/data/banks/sae131k_rl/build_stats.json")["families"]
        n = (int(fam["sae131k"]) + int(fam["sae131k_dec"])) // 2 * 2
        spec = [{"name": "acts_ctx64_2048_rl", "banks": ["/data/banks/ufw_ctx64_2048_rl_500k_v3"], "dtype": "f16", "families": ["realact_ctx64_2048"],
                 "n": n, "all_one_family": "realact_ctx64_2048", "fresh": "the paper chain's RL acts: docs [9.5M, ...), full-document forwards, ctx 64-2048"},
                {"name": "sae131k_rl", "banks": ["/data/banks/sae131k_rl"], "dtype": "f32", "families": ["sae131k", "sae131k_dec"],
                 "n": {"sae131k": n // 2, "sae131k_dec": n // 2}, "fresh": "131k L42 SAE, RL split (13,107 features; disjoint from SFT + eval)"}]
        c = modal.Function.from_name(APPS["mix"], "build").spawn(out_name=RL_POOL, spec_json=json.dumps(spec), seed=2041, threads=48,
                                                                 dense_frac=0.0, eval_cache=V3)
        d["compose_rl"] = {"call": c.object_id, "spec": spec, "n_each": n, "spawned": now()}; save(d)
        log(f"compose RL {c.object_id}: acts {n} + sae {n}")
        wait_call(c.object_id, "compose RL")
    if "rl" not in d:
        wait_call(d["sft"]["train"], "SFT train")
        final = f"/data/sft_mix/{SFT_RUN}/final"
        for _ in range(40):
            if vol_has(final):
                break
            time.sleep(60)
        save_dir = f"/data/ckpts_{RL_RUN}"
        extra = f"{RL_EXTRA} --run-name {RL_RUN} --save-dir {save_dir}"
        t = modal.Function.from_name(APPS["rl"], "train").spawn(n_rollout=2, n_trainer=6, total_steps=RL_STEPS, extra_args=extra,
                                                                pool_dir=f"/data/banks/{RL_POOL}", policy_base=final, full_param=True)
        e = evaluators(save_dir, RL_RUN, RL_STEPS)
        d["rl"] = {"train": t.object_id, "evals": e, "save": save_dir, "policy_base": final, "extra": extra, "spawned": now()}; save(d)
        log(f"RL spawned: train {t.object_id} evals {e}")
        discord(f"SFT done; RL launched ({RL_RUN}, 300 steps, ~12 h)")
    wait_call(d["rl"]["train"], "RL train")
    discord("RL finished; evaluators finishing the last checkpoints")


if __name__ == "__main__":
    main()
