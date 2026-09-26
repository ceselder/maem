"""In-distribution evals vs training compute for BOTH checkpoint chains on shared axes: Qwen3.6-27B (the paper chain; 2M-SAE data,
131k-SAE unseen eval) and its scaled-down Qwen3-8B copy (branch qwen3-8b; Karvonen 65k-SAE data, Karvonen 16k-SAE unseen eval).
Colour = stage (SFT blue, RL orange), line style = model (27B solid, 8B dashed), grey = the untrained base model of each.
Compute: the codebase's meter convention per model (plot_compute_vs_eval docstring; N from each SFT meter, scoring-pass layer
fraction 43/64 for 27B, 28/36 for 8B); each chain's RL is stacked on its own SFT final. Both chains count the 102-token shared
prompt prefix ONCE: the 27B ran with the prefix cache (its logged token counts already exclude it); the 8B ran without it (that
cache is Qwen3.6-specific), so its logged per-example / per-rollout counts include the prefix and it is subtracted here
(prefix_uncounted) -- same algorithmic accounting for both; the 8B's actually-spent compute was higher by that amount.

    python scripts/plot_indist_models.py [--metric strength|fired] [--sae unseen|trained]
      (a) held-out standalone 8-64-token windows (each SFT's own recipe), centred cosine
      (b) SAE features: unseen SAE (default) or the held-out features of the SAE each SFT's data came from (decoder directions)
    -> <report>/indist_models_<sae>_<metric>.{png,pdf} + data json
"""
import argparse
import glob
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cve", f"{HERE}/plot_compute_vs_eval.py"); cve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cve)
EVJ = os.path.expanduser("~/shared/overnight/eval_ckpt_json")
CTX = "eval/realact_ctx8_64/cos"
CHAINS = {
    "27b": dict(label="Qwen3.6-27B", ls="-", base_ls=":", prefix_uncounted=0, raw="~/shared/overnight/simple2m/flops_plot_raw_27b.json", sft_rows=8_000_000,
                score_frac=43 / 64, ctx_tags=("sft_simple2m_sft_ctx864_c", "rl_simple2m_8x2048_last16_lr5e-7_ctx864_c"),
                base={CTX: "base_qwen36_27b_ctx864_c", "*": "base_qwen36_27b_c"},
                sae={"unseen": "eval/sae/{m}", "trained": "eval/sae2m_dec/{m}"}, unseen_name="131k", trained_name="2M"),
    "8b": dict(label="Qwen3-8B", ls="--", base_ls="-.", prefix_uncounted=102, raw="~/shared/overnight/qwen3_8b/flops_plot_raw_8b.json", sft_rows=1_992_840,
               score_frac=28 / 36, ctx_tags=None, base={"*": "base_qwen3_8b"},
               sae={"unseen": "eval/sae/{m}", "trained": "eval/sae8b_dec/{m}"}, unseen_name="16k", trained_name="65k"),
}
R, EX_PER_STEP = 16384, 4096


def chain_points(c):
    raw = json.load(open(os.path.expanduser(c["raw"]))); N = float(raw["N"])
    if c["ctx_tags"]:                                   # 27B: the 8-64 family lives in the ctx864 re-score tags
        for side, tag in zip(("evals_sft", "evals_rl"), c["ctx_tags"]):
            for f in glob.glob(f"{EVJ}/{tag}/ckpt_*.json"):
                k = os.path.basename(f)[5:-5]
                if k in raw[side]:
                    raw[side][k][CTX] = json.load(open(f))["metrics"][CTX]
    sft_rows = sorted(raw["sft_rows"], key=lambda x: x["_step"]); rl_rows = sorted(raw["rl_rows"], key=lambda x: x["_step"])
    P = c["prefix_uncounted"]                           # shared prompt prefix counted once (see docstring)
    sft_fl = np.cumsum([6 * N * EX_PER_STEP * (x["tok_per_s"] / x["ex_per_s"] - P) for x in sft_rows])
    rl_fl = np.cumsum([R * N * (8 * (x["trainer/body_tokens_per_rollout"] - P) + 2 * c["score_frac"] * (x["rollout/len_mean"] + 1))
                       for x in rl_rows])
    es = {int(k): v for k, v in raw["evals_sft"].items()}; er = {int(k): v for k, v in raw["evals_rl"].items() if int(k) <= len(rl_fl)}
    sft = [(float(sft_fl[min(k, len(sft_fl)) - 1]), k, es[k]) for k in sorted(es)]
    rl = [(sft[-1][0], 0, sft[-1][2])] + [(sft[-1][0] + float(rl_fl[k - 1]), k, er[k]) for k in sorted(er)] if er else []
    return N, sft, rl


def base_metrics(c, key):
    tag = c["base"].get(key, c["base"]["*"])
    f = f"{EVJ}/{tag}/ckpt_0.json"
    return json.load(open(f))["metrics"].get(key) if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["strength", "fired"], default="strength")
    ap.add_argument("--sae", choices=["unseen", "trained"], default="unseen")
    a = ap.parse_args()
    m = "norm_act" if a.metric == "strength" else "fired"
    stem = f"indist_models_{a.sae}_{a.metric}"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 5.4))
    data, xs = {"metric": m, "sae": a.sae, "chains": {}, "accounting": "shared 102-token prompt prefix counted once for both chains"}, []
    for name, c in CHAINS.items():
        N, sft, rl = chain_points(c)
        skey = c["sae"][a.sae].format(m=m)
        cd = {"N": N, "prefix_uncounted": c["prefix_uncounted"], "sft_flops_total": sft[-1][0], "rl_flops_total": (rl[-1][0] - sft[-1][0]) if rl else 0,
              "sae_key": skey, "sft": [], "rl": [], "base": {CTX: base_metrics(c, CTX), skey: base_metrics(c, skey)}}
        for ax, key, sc in ((ax_a, CTX, 1), (ax_b, skey, 100)):
            if rl:
                ax.plot([x / 1e19 for x, _, _ in rl], [sc * e[key] for _, _, e in rl], c["ls"], marker="o", color=cve.RL_C, lw=1.8, ms=4,
                        markevery=slice(1, None), zorder=4)
            ax.plot([x / 1e19 for x, _, _ in sft], [sc * e[key] for _, _, e in sft], c["ls"], marker="o", color=cve.SFT_C, lw=1.8, ms=4, zorder=3)
            b = cd["base"][key]
            if b is not None:
                ax.axhline(sc * b, color=cve.BASE_C if hasattr(cve, "BASE_C") else "#8c8c8c", ls=c["base_ls"], lw=1.6, zorder=2)
        cd["sft"] = [{"step": k, "flops": x, CTX: e.get(CTX), skey: e.get(skey)} for x, k, e in sft]
        cd["rl"] = [{"step": k, "flops": x, CTX: e.get(CTX), skey: e.get(skey)} for x, k, e in rl]
        data["chains"][name] = cd
        xs += [x / 1e19 for x, _, _ in sft + rl]
    lo, hi = min(xs) / 1.3, max(xs) * 1.3
    ts = [t for t in cve.doubling_ticks(min(xs), hi) if t >= lo]
    for ax in (ax_a, ax_b):
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_xticks(ts, [f"{t:g}" for t in ts])
        ax.set_xlabel(r"training compute ($10^{19}$ FLOPs)"); ax.grid(True); ax.set_axisbelow(True)
    ax_a.set_title("(a) real activations, 8–64-token context"); ax_a.set_ylabel("mean centred cosine")
    ax_a.set_ylim(0.0, 0.9); ax_a.yaxis.set_major_locator(plt.MultipleLocator(0.1)); ax_a.yaxis.set_major_formatter("{x:.1f}")
    which = "unseen SAE" if a.sae == "unseen" else "trained SAE, held-out features"
    ax_b.set_title(f"(b) {which}: {a.metric}")
    ax_b.set_ylabel("target activation (% of corpus peak)" if a.metric == "strength" else "% of SAE features fired")
    top = max(v for cd in data["chains"].values() for side in ("sft", "rl") for p in cd[side] for v in [100 * p[cd["sae_key"]]])
    step = 10 if top > 40 else 5
    ax_b.set_ylim(0, step * (int(top // step) + 1)); ax_b.yaxis.set_major_locator(plt.MultipleLocator(step))
    gray = "#8c8c8c"
    ax_a.legend(handles=[Line2D([], [], color=cve.RL_C, marker="o", ms=4, lw=1.8, label="MAEM (RL, from SFT final)"),
                         Line2D([], [], color=cve.SFT_C, marker="o", ms=4, lw=1.8, label="MAEM (SFT)")] +
                        [Line2D([], [], color=gray, ls=c["base_ls"], lw=1.6, label=f"{c['label']} base (no training)") for c in CHAINS.values()],
                loc="center right", frameon=False, fontsize=10)
    ax_b.legend(handles=[Line2D([], [], color="#444444", ls=c["ls"], lw=1.8,
                                label=f"{c['label']}" + (f" ({c['unseen_name' if a.sae == 'unseen' else 'trained_name']}-feature SAE)"))
                         for c in CHAINS.values()], frameon=False, fontsize=10,
                **({"loc": "lower left", "bbox_to_anchor": (0.0, 0.2)} if (a.sae, a.metric) == ("unseen", "fired") else
                   {"loc": "upper left" if (a.sae, a.metric) == ("unseen", "strength") else "center"}))
    fig.tight_layout(w_pad=2.0)
    os.makedirs(f"{cve.OUT}/data", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{cve.OUT}/{stem}.{ext}", dpi=150)
    json.dump(data, open(f"{cve.OUT}/data/{stem}.json", "w"), indent=1)
    print("wrote", f"{cve.OUT}/{stem}.png + .pdf + data/{stem}.json")


if __name__ == "__main__":
    main()
