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
SFT_C, RL_C = "#2a78d6", "#eb6834"                       # validated categorical slots 1-2 (dataviz palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e4df", "#fcfcfb"
R, EX_PER_STEP, SCORE_LAYER_FRAC = 16384, 4096, 43 / 64
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


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
    metrics = [("eval/mean_all", "Held-out inversion fidelity\n(mean centred cosine, 10 direction families)",
                "RL lifts held-out fidelity; more SFT lowers it"),
               ("eval/sae/fired", "Fraction of 512 held-out features fired\n(131k-feature SAE never trained on)",
                "RL fires more of an unseen SAE's features")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0))
    data = {"convention": __doc__.split("\n\n")[0], "N_params": N, "sft_total_flops": SFT_TOTAL, "rl_total_flops": float(rl_cum[-1]),
            "rl_to_sft_compute_ratio": float(rl_cum[-1] / SFT_TOTAL), "metrics": {}}
    for ax, (key, ylab, title) in zip(axes, metrics):
        sx, sy = [p[0] for p in sft_pts], [p[1][key] for p in sft_pts]
        rx, ry = [p[0] for p in rl_pts], [p[1][key] for p in rl_pts]
        ax.plot(sx, sy, "-o", color=SFT_C, lw=2, ms=7, mec=SURF, mew=1.5, label="SFT (1M to 8M examples)", zorder=3)
        ax.plot(rx, ry, "-s", color=RL_C, lw=2, ms=7, mec=SURF, mew=1.5, label="RL from the SFT final (25 to 300 steps)", zorder=4)
        for x, y, lab, dy, ha in ((sx[0], sy[0], f"{sy[0]:.2f}", 10, "center"), (sx[-1], sy[-1], f"{sy[-1]:.2f}", -18, "center"),
                                  (rx[-1], ry[-1], f"{ry[-1]:.2f}", 10, "center")):
            ax.annotate(lab, (x, y), xytext=(0, dy), textcoords="offset points", ha=ha, fontsize=11, color=INK)
        ax.text(sx[1], sy[1], "  SFT", color=INK2, fontsize=12, va="bottom", ha="left")
        ax.text(rx[-2], ry[-2], "RL  ", color=INK2, fontsize=12, va="bottom", ha="right")
        ax.set_xscale("log"); ax.grid(color=GRID, lw=0.8, which="major"); ax.set_axisbelow(True)
        ax.set_xlabel("Cumulative training compute (FLOPs)"); ax.set_ylabel(ylab); ax.set_title(title, loc="left", color=INK, fontsize=13)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        lo, hi = min(sy + ry), max(sy + ry); ax.set_ylim(lo - 0.04, hi + 0.04)
        data["metrics"][key] = {"sft": [{"step": k, "flops": x, "value": v[key]} for k, (x, v) in zip(sorted(es), sft_pts)],
                                "rl": [{"step": s, "flops": x, "value": v[key]} for s, (x, v) in zip([0] + sorted(er), rl_pts)]}
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("RL compute raises held-out fidelity from 0.42 to 0.60 while SFT compute lowers it"
                 f"\nQwen3.6-27B activation inverter, 9 SFT and 7 RL checkpoints; RL used {rl_cum[-1] / SFT_TOTAL:.1f}x the SFT compute",
                 x=0.01, y=0.985, ha="left", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.subplots_adjust(top=0.83)
    os.makedirs(f"{OUT}/data", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/compute_vs_eval.{ext}", dpi=150)
    json.dump(data, open(f"{OUT}/data/compute_vs_eval.json", "w"), indent=1)
    print(f"SFT total {SFT_TOTAL:.3e} FLOPs | RL total {rl_cum[-1]:.3e} FLOPs | RL/SFT {rl_cum[-1] / SFT_TOTAL:.2f}")
    print("wrote", f"{OUT}/compute_vs_eval.png", "+ .pdf + data/compute_vs_eval.json")


if __name__ == "__main__":
    main()
