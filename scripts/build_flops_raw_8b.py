"""Raw per-step compute logs + per-checkpoint evals for the Qwen3-8B chain (branch qwen3-8b) in the schema plot_compute_vs_eval.budgets
reads (the 27B file is ~/shared/overnight/simple2m/flops_plot_raw_27b.json). N = params implied by the SFT FLOP meter
(tflops / (6 * tok_per_s), median over steps), the same convention as the 27B. Re-run as RL checkpoints are scored.

    python scripts/build_flops_raw_8b.py     # -> ~/shared/overnight/qwen3_8b/flops_plot_raw_8b.json
"""
import glob
import json
import os

import numpy as np
import wandb

OUT = os.path.expanduser("~/shared/overnight/qwen3_8b/flops_plot_raw_8b.json")
EVJ = os.path.expanduser("~/shared/overnight/eval_ckpt_json")
PROJECT = "celestedeschamphelaere-personal/maxact-fast"
SFT_RUN, RL_NAME = "18aha7t4", "qwen3_8b_rl_last16_lr5e-7"
SFT_TAG, RL_TAG, BASE_TAG = "sft_qwen3_8b_sft_c", "qwen3_8b_rl_last16_lr5e-7_c", "base_qwen3_8b"
SFT_KEYS = ["_step", "ex_per_s", "tok_per_s", "tflops"]
RL_KEYS = ["_step", "rollout/len_mean", "trainer/body_tokens_per_rollout", "trainer/real_tokens_per_rollout"]


def evals(tag):
    return {os.path.basename(f)[5:-5]: json.load(open(f))["metrics"] for f in glob.glob(f"{EVJ}/{tag}/ckpt_*.json")}


def main():
    api = wandb.Api()
    sft = [r for r in api.run(f"{PROJECT}/{SFT_RUN}").scan_history(keys=SFT_KEYS) if r.get("tok_per_s")]
    rl_runs = [r for r in api.runs(PROJECT, filters={"display_name": RL_NAME}, order="-created_at")]
    rl = [r for r in rl_runs[0].scan_history(keys=RL_KEYS) if r.get("trainer/body_tokens_per_rollout")] if rl_runs else []
    N = float(np.median([r["tflops"] * 1e12 / (6 * r["tok_per_s"]) for r in sft]))
    raw = {"N": N, "sft_rows": sft, "rl_rows": rl, "evals_sft": evals(SFT_TAG), "evals_rl": evals(RL_TAG),
           "base": json.load(open(f"{EVJ}/{BASE_TAG}/ckpt_0.json"))["metrics"], "sft_run": SFT_RUN, "rl_run": rl_runs[0].id if rl_runs else None}
    json.dump(raw, open(OUT, "w"))
    print(f"N={N / 1e9:.3f}B | sft rows {len(sft)} | rl rows {len(rl)} | evals sft {sorted(map(int, raw['evals_sft']))} rl {sorted(map(int, raw['evals_rl']))}")


if __name__ == "__main__":
    main()
