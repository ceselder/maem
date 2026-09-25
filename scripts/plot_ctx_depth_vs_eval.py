"""Real-activation inversion vs training compute, split by how much context the activation had: held-out standalone 8-64-token
windows (the SFT data's own distribution; eval/modal_build_ctx864_eval.py) vs real activations at positions 16-511, 512-2048,
2048-8192. Same checkpoint chain, FLOPs convention and style as plot_compute_vs_eval.py (budgets / ticks imported from it).
Eval scores: centered scorer, best of 4 at T=1, 512 directions per family; the 8-64 family comes from the ctx864 re-score tags.

    python scripts/plot_ctx_depth_vs_eval.py     # -> <report>/ctx_depth_vs_eval.{png,pdf} + data/ctx_depth_vs_eval.json
"""
import glob
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cve", f"{HERE}/plot_compute_vs_eval.py"); cve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cve)
EVJ = os.path.expanduser("~/shared/overnight/eval_ckpt_json")
TAGS = {"evals_sft": "sft_simple2m_sft_ctx864_c", "evals_rl": "rl_simple2m_8x2048_last16_lr5e-7_ctx864_c"}
NEW = "eval/realact_ctx8_64/cos"
PANELS = [(NEW, "(a) 8–64-token windows (SFT's distribution)"), ("eval/realact/cos", "(b) positions 16–511"),
          ("eval/realact_mid/cos", "(c) positions 512–2048"), ("eval/realact_long/cos", "(d) positions 2048–8192")]


def main():
    raw = json.load(open(cve.RAW))
    for side, tag in TAGS.items():                      # merge the 8-64 family into the per-checkpoint eval dicts
        for f in glob.glob(f"{EVJ}/{tag}/ckpt_*.json"):
            k = os.path.basename(f)[5:-5]
            raw[side][k][NEW] = json.load(open(f))["metrics"][NEW]
    N, sft, rl, _ = cve.budgets(raw)
    off = sft[-1]["flops"]
    sft_pts = [(p["flops"] / 1e19, p) for p in sft]; rl_pts = [((off + p["flops"]) / 1e19, p) for p in rl]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6), sharex=True, sharey=True)
    data = {"x": "cumulative training compute (1e19 FLOPs), RL stacked on the SFT final", "eval": "centered cosine, best of 4 at T=1, n=512",
            "families": {}}
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.plot([x for x, _ in rl_pts], [p["evals"][key] for _, p in rl_pts], "-o", color=cve.RL_C, lw=1.8, ms=4, markevery=slice(1, None),
                label="MAEM (RL, from SFT final)", zorder=4)
        ax.plot([x for x, _ in sft_pts], [p["evals"][key] for _, p in sft_pts], "-o", color=cve.SFT_C, lw=1.8, ms=4, label="MAEM (SFT)", zorder=3)
        ax.set_title(title); ax.grid(True); ax.set_axisbelow(True)
        data["families"][key] = {s: [{"step": p["step"], "flops": p["flops"], "value": p["evals"][key]} for _, p in pts]
                                 for s, pts in (("sft", sft_pts), ("rl", rl_pts))}
    xs = [x for x, _ in sft_pts + rl_pts]; lo, hi = min(xs) / 1.3, max(xs) * 1.3
    ts = [t for t in cve.doubling_ticks(min(xs), hi) if t >= lo]
    for ax in axes.flat:
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_xticks(ts, [f"{t:g}" for t in ts])
    for ax in axes[1]:
        ax.set_xlabel(r"training compute ($10^{19}$ FLOPs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean centred cosine")
    axes[0, 0].yaxis.set_major_locator(plt.MultipleLocator(0.1)); axes[0, 0].yaxis.set_major_formatter("{x:.1f}")
    axes[0, 0].set_ylim(0.35, 0.85)
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    os.makedirs(f"{cve.OUT}/data", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{cve.OUT}/ctx_depth_vs_eval.{ext}", dpi=150)
    json.dump(data, open(f"{cve.OUT}/data/ctx_depth_vs_eval.json", "w"), indent=1)
    print("wrote", f"{cve.OUT}/ctx_depth_vs_eval.png + .pdf + data/ctx_depth_vs_eval.json")


if __name__ == "__main__":
    main()
