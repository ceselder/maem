"""Held-out evals vs cumulative training compute for the paper's checkpoint chain: 9 SFT checkpoints, then 7 RL checkpoints
of the RL run initialised from the SFT final. Compute follows the codebase's own meter convention (mxf/mfu.py): 6N FLOPs per
trained token (forward + backward), 2N per forward-only token, times the layer fraction for truncated forwards. Token counts
come from each run's per-step logs, pad excluded, the shared prompt prefix counted once (prefix caching).

  SFT step:  6 N * 4096 examples * (tokens per example)
  RL step:   16384 rollouts * N * [ 6*body (trainer fwd+bwd) + 2*body (vLLM generation) + 2*(43/64)*(len+1) (layer-42 scoring pass) ]

    python scripts/plot_compute_vs_eval.py     # -> <report>/compute_vs_eval.{png,pdf} + data/compute_vs_eval.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser(os.environ.get("MAEMM_SIMPLE2M_REPORT", "~/shared/reports/maemm-simple2m-chain"))
RAW = "/tmp/flops_plot_raw.json"
SFT_C, RL_C = "#3576c9", "#d06c38"                       # same blue/orange as the paper's other figures (validated CVD pair)
R, EX_PER_STEP, SCORE_LAYER_FRAC = 16384, 4096, 43 / 64
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.linewidth": 0.8, "grid.color": "#b0b0b0", "grid.alpha": 0.35,
                     "grid.linewidth": 0.8, "mathtext.default": "regular"})
XTICKS = [(5e18, r"$5{\times}10^{18}$"), (1e19, r"$10^{19}$"), (2e19, r"$2{\times}10^{19}$"), (5e19, r"$5{\times}10^{19}$"),
          (1e20, r"$10^{20}$")]


def main():
    raw = json.load(open(RAW)); N = float(raw["N"])
    sft_rows = sorted(raw["sft_rows"], key=lambda x: x["_step"]); rl_rows = sorted(raw["rl_rows"], key=lambda x: x["_step"])
    sft_step = np.array([6 * N * EX_PER_STEP * x["tok_per_s"] / x["ex_per_s"] for x in sft_rows])
    rl_step = np.array([R * N * (8 * x["trainer/body_tokens_per_rollout"] + 2 * SCORE_LAYER_FRAC * (x["rollout/len_mean"] + 1)) for x in rl_rows])
    sft_cum, rl_cum = np.cumsum(sft_step), np.cumsum(rl_step)
    SFT_TOTAL = float(sft_cum[-1])
    es = {int(k): v for k, v in raw["evals_sft"].items()}; er = {int(k): v for k, v in raw["evals_rl"].items()}
    # checkpoint step k = k optimizer steps done; the SFT final (1954) is every logged step
    sft_pts = [(float(sft_cum[min(k, len(sft_cum)) - 1]), es[k]) for k in sorted(es)]
    rl_pts = [(SFT_TOTAL, es[max(es)])] + [(SFT_TOTAL + float(rl_cum[k - 1]), er[k]) for k in sorted(er)]   # RL starts at the SFT final
    metrics = [("eval/mean_all", 1, "(a) all direction families: centred cosine", "mean cosine", 0.05, "{x:.2f}"),
               ("eval/sae/fired", 100, "(b) SAE features: target fires", "% of SAE features fired", 5, "{x:.0f}")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4))
    data = {"convention": __doc__.split("\n\n")[0], "N_params": N, "sft_total_flops": SFT_TOTAL, "rl_total_flops": float(rl_cum[-1]),
            "rl_to_sft_compute_ratio": float(rl_cum[-1] / SFT_TOTAL), "eval": "held-out, best of 4 samples at T=1; fired plotted x100 as %",
            "metrics": {}}
    for ax, (key, scale, title, ylab, ystep, yfmt) in zip(axes, metrics):
        sx, sy = [p[0] for p in sft_pts], [scale * p[1][key] for p in sft_pts]
        rx, ry = [p[0] for p in rl_pts], [scale * p[1][key] for p in rl_pts]
        ax.plot(rx, ry, "-o", color=RL_C, lw=1.8, ms=4, markevery=slice(1, None), label="MAEM (RL, from SFT final)", zorder=4)
        ax.plot(sx, sy, "-o", color=SFT_C, lw=1.8, ms=4, label="MAEM (SFT)", zorder=3)
        ax.set_xscale("log"); ax.set_xticks([t for t, _ in XTICKS], [l for _, l in XTICKS]); ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(4.2e18, 1.4e20); ax.grid(True); ax.set_axisbelow(True)
        ax.set_xlabel("training compute (FLOPs)"); ax.set_ylabel(ylab); ax.set_title(title)
        lo, hi = min(sy + ry), max(sy + ry); pad = 0.12 * (hi - lo); ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(plt.MultipleLocator(ystep)); ax.yaxis.set_major_formatter(yfmt)
        data["metrics"][key] = {"sft": [{"step": k, "flops": x, "value": v[key]} for k, (x, v) in zip(sorted(es), sft_pts)],
                                "rl": [{"step": s, "flops": x, "value": v[key]} for s, (x, v) in zip([0] + sorted(er), rl_pts)]}
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout(w_pad=2.0)
    os.makedirs(f"{OUT}/data", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/compute_vs_eval.{ext}", dpi=150)
    json.dump(data, open(f"{OUT}/data/compute_vs_eval.json", "w"), indent=1)
    print(f"SFT total {SFT_TOTAL:.3e} FLOPs | RL total {rl_cum[-1]:.3e} FLOPs | RL/SFT {rl_cum[-1] / SFT_TOTAL:.2f}")
    print("wrote", f"{OUT}/compute_vs_eval.png", "+ .pdf + data/compute_vs_eval.json")


if __name__ == "__main__":
    main()
