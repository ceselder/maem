"""Score the simple2m SFT + RL (last16, lr 5e-7) checkpoints on the held-out SFT-distribution family (standalone 8-64-token
windows from eval-head docs; eval/modal_build_ctx864_eval.py), centered scorer, one B200 per checkpoint, in parallel.
Results: /data/eval_ckpt/<tag>/ckpt_<k>.json (eval/realact_ctx8_64/cos, eval/realact/cos sanity, eval/sae/*);
call ids in ~/shared/overnight/simple2m/ids.json["ctx864_eval"]["calls"].

    python scripts/launchers/spawn_ctx864_eval.py
"""
import json
import os
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
APP = os.environ.get("RESCORE_APP", "maemm-eval-ckpt-s2m-c")
CACHE = "/data/eval_universal_ho/eval_sets_ctx864.pt"
EXTRA = f"--full-model --no-wandb --dump-per-dir --eval-cache {CACHE} --no-extra-evals"   # per-dir -> exact SE + paired diffs
IDS = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
RUNS = [("/data/sft_mix/simple2m_sft", "sft_simple2m_sft_ctx864_c", 1954, [244, 488, 732, 976, 1220, 1464, 1708, 1952, 1954]),
        ("/data/ckpts_rl_simple2m_8x2048_last16_lr5e-7", "rl_simple2m_8x2048_last16_lr5e-7_ctx864_c", 300, [25, 50, 100, 150, 200, 250, 300])]


def main():
    calls = {}
    daemon = modal.Function.from_name(APP, "daemon")
    for ckpt_dir, tag, final_step, steps in RUNS:
        for s in steps:
            fc = daemon.spawn(ckpt_dir=ckpt_dir, tag=tag, once=True, only_step=s, final_step=final_step, extra_args=EXTRA)
            calls[f"{tag}/step_{s}"] = fc.object_id
            print(f"{tag} step {s}: {fc.object_id}")
    d = json.load(open(IDS)); d.setdefault("ctx864_eval", {})
    d["ctx864_eval"].update({"eval_app": APP, "extra_args": EXTRA, "calls": calls,
                             "evals_spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    json.dump(d, open(IDS, "w"), indent=1)


if __name__ == "__main__":
    main()
