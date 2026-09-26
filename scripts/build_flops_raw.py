"""Raw per-step compute logs + per-checkpoint evals of a checkpoint chain in the schema plot_indist_models reads (the paper 27B chain's
file is ~/shared/overnight/simple2m/flops_plot_raw_27b.json). N = params implied by the SFT FLOP meter (tflops / (6 * tok_per_s),
median over steps). Re-run as RL checkpoints are scored.

    python scripts/build_flops_raw.py --chain 8b       # Qwen3-8B chain (branch qwen3-8b)       -> ~/shared/overnight/qwen3_8b/flops_plot_raw_8b.json
    python scripts/build_flops_raw.py --chain 27b131k  # 27B trained on 131k-SAE data (sae131k-indist) -> ~/shared/overnight/sae131k_indist/flops_plot_raw_27b131k.json
"""
import argparse
import glob
import json
import os

import numpy as np
import wandb

EVJ = os.path.expanduser("~/shared/overnight/eval_ckpt_json")
PROJECT = "celestedeschamphelaere-personal/maxact-fast"
CHAINS = {  # sft: wandb run id or display name; ctx tags (27B chains): the 8-64-token family lives in separate re-score tags
    "8b": dict(out="~/shared/overnight/qwen3_8b/flops_plot_raw_8b.json", sft="18aha7t4", rl="qwen3_8b_rl_last16_lr5e-7",
               tags=("sft_qwen3_8b_sft_c", "qwen3_8b_rl_last16_lr5e-7_c"), ctx_tags=None, base="base_qwen3_8b"),
    "27b131k": dict(out="~/shared/overnight/sae131k_indist/flops_plot_raw_27b131k.json", sft="sae131k_sft", rl="rl_sae131k_8x2048_last16_lr5e-7",
                    tags=("sft_sae131k_sft_c", "rl_sae131k_8x2048_last16_lr5e-7_c"),
                    ctx_tags=("sft_sae131k_sft_ctx864_c", "rl_sae131k_8x2048_last16_lr5e-7_ctx864_c"), base="base_qwen36_27b_c"),
}
SFT_KEYS = ["_step", "ex_per_s", "tok_per_s", "tflops"]
RL_KEYS = ["_step", "rollout/len_mean", "trainer/body_tokens_per_rollout", "trainer/real_tokens_per_rollout"]


def evals(tag):
    return {os.path.basename(f)[5:-5]: json.load(open(f))["metrics"] for f in glob.glob(f"{EVJ}/{tag}/ckpt_*.json")}


def run_of(api, key):
    if len(key) == 8 and key.isalnum():                 # a wandb run id; otherwise a display name
        try:
            return api.run(f"{PROJECT}/{key}")
        except Exception:
            pass
    rs = [r for r in api.runs(PROJECT, filters={"display_name": key}, order="-created_at")]
    return rs[0] if rs else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chain", choices=list(CHAINS), default="8b"); a = ap.parse_args()
    c = CHAINS[a.chain]; OUT = os.path.expanduser(c["out"])
    api = wandb.Api()
    sft_run, rl_run = run_of(api, c["sft"]), run_of(api, c["rl"])
    sft = [r for r in sft_run.scan_history(keys=SFT_KEYS) if r.get("tok_per_s")]
    rl = [r for r in rl_run.scan_history(keys=RL_KEYS) if r.get("trainer/body_tokens_per_rollout")] if rl_run else []
    N = float(np.median([r["tflops"] * 1e12 / (6 * r["tok_per_s"]) for r in sft]))
    es, er = evals(c["tags"][0]), evals(c["tags"][1])
    if c["ctx_tags"]:
        for d, tag in zip((es, er), c["ctx_tags"]):
            for k, m in evals(tag).items():
                if k in d and "eval/realact_ctx8_64/cos" in m:
                    d[k]["eval/realact_ctx8_64/cos"] = m["eval/realact_ctx8_64/cos"]
    raw = {"N": N, "sft_rows": sft, "rl_rows": rl, "evals_sft": es, "evals_rl": er,
           "base": json.load(open(f"{EVJ}/{c['base']}/ckpt_0.json"))["metrics"], "sft_run": sft_run.id, "rl_run": rl_run.id if rl_run else None}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(raw, open(OUT, "w"))
    print(f"N={N / 1e9:.3f}B | sft rows {len(sft)} | rl rows {len(rl)} | evals sft {sorted(map(int, raw['evals_sft']))} rl {sorted(map(int, raw['evals_rl']))}")


if __name__ == "__main__":
    main()
