"""Held-out evals vs training budget for the paper's checkpoint chain: 9 SFT checkpoints, then 7 RL checkpoints of the RL run
initialised from the SFT final. Compute follows the codebase's own meter convention (mxf/mfu.py): 6N FLOPs per trained token
(forward + backward), 2N per forward-only token, times the layer fraction for truncated forwards. Token counts come from each
run's per-step logs, pad excluded, the shared prompt prefix counted once (prefix caching).

  SFT step:  6 N * 4096 examples * (tokens per example)
  RL step:   16384 rollouts * N * [ 6*body (trainer fwd+bwd) + 2*body (vLLM generation) + 2*(43/64)*(len+1) (layer-42 scoring pass) ]

Budget axes (--x): flops (as above), directions (training directions seen: SFT 4096 per step, 8,000,000 total, half real
activations / half SAE features; RL 2048 prompts per step with 8 rollouts each, pool also half real activations), steps
(optimizer steps). --scale log|linear. --rl-origin sft_end stacks RL on top of the SFT budget; zero counts each stage from 0
(RL's step-0 point = the SFT final; on a log axis it cannot be drawn, so the RL line starts at step 25).

    python scripts/plot_compute_vs_eval.py          # -> <report>/compute_vs_eval.{png,pdf} + data/compute_vs_eval.json (flops, log, stacked)
    python scripts/plot_compute_vs_eval.py --all    # -> every x/scale/origin combination as eval_vs_<x>_<scale>[_rl0].*
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser(os.environ.get("MAEMM_SIMPLE2M_REPORT", "~/shared/reports/maemm-simple2m-chain"))
RAW = "/tmp/flops_plot_raw.json"
SFT_C, RL_C = "#3576c9", "#d06c38"                       # same blue/orange as the paper's other figures (validated CVD pair)
R, EX_PER_STEP, RL_DIRS_PER_STEP, SFT_ROWS, SCORE_LAYER_FRAC = 16384, 4096, 2048, 8_000_000, 43 / 64
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.linewidth": 0.8, "grid.color": "#b0b0b0", "grid.alpha": 0.35,
                     "grid.linewidth": 0.8, "mathtext.default": "regular"})
METRICS = [("eval/realact/cos", 1, "(a) real activations: centred cosine", "mean cosine", 0.05, "{x:.2f}"),
           ("eval/sae/fired", 100, "(b) SAE features: target fires", "% of SAE features fired", 5, "{x:.0f}")]
# x key -> axis name per origin, linear-axis divisor, linear unit suffix
XLAB = {"flops": {"sft_end": "training compute", "zero": "compute spent in each stage"},
        "directions": {"sft_end": "training directions seen", "zero": "directions seen in each stage"},
        "steps": {"sft_end": "optimizer steps", "zero": "optimizer steps in each stage"}}
LIN_DIV = {"flops": (1e19, r" ($10^{19}$ FLOPs)"), "directions": (1e6, " (millions)"), "steps": (1, "")}
LOG_DIV = {"flops": (1e19, r" ($10^{19}$ FLOPs)"), "directions": (1, ""), "steps": (1, "")}


def doubling_ticks(lo, hi):
    """Evenly spaced log ticks: base * 2^k, base = largest {1, 2.5, 5} x 10^e <= lo; every other one if more than 7."""
    base = max(m * 10 ** e for e in range(-3, 10) for m in (1, 2.5, 5) if m * 10 ** e <= lo)
    ts = [base * 2 ** k for k in range(40) if base * 2 ** k <= hi]
    return ts[::2] if len(ts) > 7 else ts


def short(v):
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"


def budgets(raw):
    N = float(raw["N"])
    sft_rows = sorted(raw["sft_rows"], key=lambda x: x["_step"]); rl_rows = sorted(raw["rl_rows"], key=lambda x: x["_step"])
    sft_fl = np.cumsum([6 * N * EX_PER_STEP * x["tok_per_s"] / x["ex_per_s"] for x in sft_rows])
    rl_fl = np.cumsum([R * N * (8 * x["trainer/body_tokens_per_rollout"] + 2 * SCORE_LAYER_FRAC * (x["rollout/len_mean"] + 1)) for x in rl_rows])
    es = {int(k): v for k, v in raw["evals_sft"].items()}; er = {int(k): v for k, v in raw["evals_rl"].items()}
    # checkpoint step k = k optimizer steps done; the SFT final (1954) is every logged step, the last one partial (8M rows total)
    sft = [{"step": k, "steps": k, "flops": float(sft_fl[min(k, len(sft_fl)) - 1]), "directions": min(k * EX_PER_STEP, SFT_ROWS),
            "evals": es[k]} for k in sorted(es)]
    rl = [{"step": 0, "steps": 0, "flops": 0.0, "directions": 0, "evals": es[max(es)]}] + \
         [{"step": k, "steps": k, "flops": float(rl_fl[k - 1]), "directions": k * RL_DIRS_PER_STEP, "evals": er[k]} for k in sorted(er)]
    return N, sft, rl, float(rl_fl[-1])


def plot(N, sft, rl, rl_total_flops, x, scale, origin):
    stem = "compute_vs_eval" if (x, scale, origin) == ("flops", "log", "sft_end") else f"eval_vs_{x}_{scale}" + ("_rl0" if origin == "zero" else "")
    off = sft[-1][x] if origin == "sft_end" else 0
    sft_pts = [(p[x], p) for p in sft]; rl_pts = [(off + p[x], p) for p in rl]
    if scale == "log" and origin == "zero":
        rl_pts = rl_pts[1:]                                   # RL step 0 sits at x = 0: not drawable on a log axis
    div, unit = (LIN_DIV if scale == "linear" else LOG_DIV)[x]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4))
    data = {"convention": __doc__.split("\n\n")[0], "x": x, "scale": scale, "rl_origin": origin, "N_params": N,
            "sft_total_flops": sft[-1]["flops"], "rl_total_flops": rl_total_flops, "rl_to_sft_compute_ratio": rl_total_flops / sft[-1]["flops"],
            "eval": "held-out, best of 4 samples at T=1; fired plotted x100 as %", "metrics": {}}
    for ax, (key, sc, title, ylab, ystep, yfmt) in zip(axes, METRICS):
        sy = [sc * p["evals"][key] for _, p in sft_pts]; ry = [sc * p["evals"][key] for _, p in rl_pts]
        # stacked: RL's first point IS the SFT final dot, so it carries no marker of its own
        ax.plot([v / div for v, _ in rl_pts], ry, "-o", color=RL_C, lw=1.8, ms=4, label="MAEM (RL, from SFT final)", zorder=4,
                markevery=slice(1, None) if origin == "sft_end" else None)
        ax.plot([v / div for v, _ in sft_pts], sy, "-o", color=SFT_C, lw=1.8, ms=4, zorder=3,
                label="MAEM (SFT)" if origin == "sft_end" else "MAEM (SFT, from base model)")
        xs = [v / div for v, _ in rl_pts + sft_pts]
        if scale == "log":
            lo, hi = min(xs) / 1.3, max(xs) * 1.3; ts = [t for t in doubling_ticks(min(xs), hi) if t >= lo]
            ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_xticks(ts, [short(t) if x == "directions" else f"{t:g}" for t in ts])
        else:
            lo = 0 if origin == "zero" else min(xs); span = max(xs) - lo; ax.set_xlim(lo - 0.04 * span, max(xs) + 0.04 * span)
        ax.grid(True); ax.set_axisbelow(True); ax.set_xlabel(XLAB[x][origin] + unit); ax.set_ylabel(ylab); ax.set_title(title)
        lo, hi = min(sy + ry), max(sy + ry); pad = 0.12 * (hi - lo); ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(plt.MultipleLocator(ystep)); ax.yaxis.set_major_formatter(yfmt)
        data["metrics"][key] = {s: [{"step": p["step"], "flops": p["flops"], "directions": p["directions"], "x": v, "value": p["evals"][key]}
                                    for v, p in pts] for s, pts in (("sft", sft_pts), ("rl", rl_pts))}
    axes[0].legend(loc="best", frameon=False)
    fig.tight_layout(w_pad=2.0)
    os.makedirs(f"{OUT}/data", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=150)
    plt.close(fig)
    json.dump(data, open(f"{OUT}/data/{stem}.json", "w"), indent=1)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", choices=list(XLAB), default="flops")
    ap.add_argument("--scale", choices=["log", "linear"], default="log")
    ap.add_argument("--rl-origin", choices=["sft_end", "zero"], default="sft_end")
    ap.add_argument("--all", action="store_true", help="render every x/scale/origin combination")
    a = ap.parse_args()
    N, sft, rl, rl_total = budgets(json.load(open(RAW)))
    combos = itertools.product(XLAB, ("log", "linear"), ("sft_end", "zero")) if a.all else [(a.x, a.scale, a.rl_origin)]
    for c in combos:
        print("wrote", f"{OUT}/{plot(N, sft, rl, rl_total, *c)}.png", "+ .pdf + data json")
    print(f"SFT total {sft[-1]['flops']:.3e} FLOPs | RL total {rl_total:.3e} FLOPs | RL/SFT {rl_total / sft[-1]['flops']:.2f}")


if __name__ == "__main__":
    main()
