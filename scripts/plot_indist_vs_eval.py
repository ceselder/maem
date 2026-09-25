"""In-distribution evals vs training compute: (a) real activations from held-out standalone 8-64-token windows (the SFT data's own
recipe; eval/modal_build_ctx864_eval.py), (b) held-out features of the SAE the SFT data was built from (the 2,097,152-feature
layer-42 SAE: 512 features of its eval split, never in any training or RL bank; encoder and decoder directions, % whose target
feature fires above the SAE's gate). Same checkpoint chain, FLOPs convention and style as plot_compute_vs_eval.py.

    python scripts/plot_indist_vs_eval.py                 # (b) = % fired        -> <report>/indist_vs_eval.{png,pdf} + data json
    python scripts/plot_indist_vs_eval.py --strength      # (b) = activation strength: mean over features of best-of-4 max-token
                                                          #       target activation / that feature's corpus peak (1.0B-token scan)
    add --zoom for panel (a) on 0.675-0.725 (base-model line then off-scale)       -> <report>/indist_vs_eval[_strength][_zoom].*
"""
import argparse
import glob
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cve", f"{HERE}/plot_compute_vs_eval.py"); cve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cve)
EVJ = os.path.expanduser("~/shared/overnight/eval_ckpt_json")
TAGS = {"evals_sft": "sft_simple2m_sft_ctx864_c", "evals_rl": "rl_simple2m_8x2048_last16_lr5e-7_ctx864_c"}
CTX = "eval/realact_ctx8_64/cos"
STYLES = [("dec", "-", "decoder directions"), ("enc", "--", "encoder directions")]
BASE_C = "#8c8c8c"                                   # untrained base model: gray dotted reference line (as in the paper figures)
BASE_TAGS = {CTX: "base_qwen36_27b_ctx864_c", "sae2m": "base_qwen36_27b_c"}   # the v3 base eval also carries the 131k "sae" family   # step 0 = Qwen3.6-27B, no MAEM training


def base_value(key):
    tag = BASE_TAGS[CTX] if key == CTX else BASE_TAGS["sae2m"]
    f = f"{EVJ}/{tag}/ckpt_0.json"
    return json.load(open(f))["metrics"].get(key) if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--strength", action="store_true")
    ap.add_argument("--zoom", action="store_true", help="panel (a) zoomed to 0.675-0.725 (the base-model line is then off-scale, omitted)")
    ap.add_argument("--sae", choices=["trained", "unseen"], default="trained",
                    help="(b): held-out features of the SAE the SFT data came from (2M; enc + dec) or the 131k SAE never trained on (enc)")
    a = ap.parse_args()
    metric, stem = ("norm_act", "indist_vs_eval_strength") if a.strength else ("fired", "indist_vs_eval")
    stem += ("_unseen" if a.sae == "unseen" else "") + ("_zoom" if a.zoom else "")
    SAE = ([(f"eval/sae/{metric}", "-", None)] if a.sae == "unseen" else
           [(f"eval/sae2m_{d}/{metric}", ls, lab) for d, ls, lab in STYLES])
    raw = json.load(open(cve.RAW))
    for side, tag in TAGS.items():                      # merge the 8-64 family into the per-checkpoint eval dicts
        for f in glob.glob(f"{EVJ}/{tag}/ckpt_*.json"):
            raw[side][os.path.basename(f)[5:-5]][CTX] = json.load(open(f))["metrics"][CTX]
    N, sft, rl, _ = cve.budgets(raw)
    off = sft[-1]["flops"]
    sft_pts = [(p["flops"] / 1e19, p) for p in sft]; rl_pts = [((off + p["flops"]) / 1e19, p) for p in rl]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 5.4))
    data = {"x": "cumulative training compute (1e19 FLOPs), RL stacked on the SFT final", "eval": "best of 4 at T=1, n=512 per family",
            "families": {}}

    def line(ax, key, scale, ls, label_rl=None, label_sft=None):
        ax.plot([x for x, _ in rl_pts], [scale * p["evals"][key] for _, p in rl_pts], ls, marker="o", color=cve.RL_C, lw=1.8, ms=4,
                markevery=slice(1, None), label=label_rl, zorder=4)
        ax.plot([x for x, _ in sft_pts], [scale * p["evals"][key] for _, p in sft_pts], ls, marker="o", color=cve.SFT_C, lw=1.8, ms=4,
                label=label_sft, zorder=3)
        data["families"][key] = {s: [{"step": p["step"], "flops": p["flops"], "value": p["evals"][key]} for _, p in pts]
                                 for s, pts in (("sft", sft_pts), ("rl", rl_pts))}

    line(ax_a, CTX, 1, "-", "MAEM (RL, from SFT final)", "MAEM (SFT)")
    ax_a.set_title("(a) real activations, 8–64-token context"); ax_a.set_ylabel("mean centred cosine")
    b = base_value(CTX)
    data["base_model"] = {CTX: b}
    if a.zoom:
        ax_a.set_ylim(0.675, 0.725); ax_a.yaxis.set_major_locator(plt.MultipleLocator(0.01)); ax_a.yaxis.set_major_formatter("{x:.2f}")
    else:
        if b is not None:
            ax_a.axhline(b, color=BASE_C, ls=":", lw=1.8, label="base model (no training)", zorder=2)
        ax_a.set_ylim(0.1, 0.8); ax_a.yaxis.set_major_locator(plt.MultipleLocator(0.1)); ax_a.yaxis.set_major_formatter("{x:.1f}")
    ax_a.legend(loc="lower right" if a.zoom else "center right", frameon=False)
    for key, ls, _ in SAE:
        line(ax_b, key, 100, ls)
        bv = base_value(key); data["base_model"][key] = bv
        if bv is not None:
            ax_b.axhline(100 * bv, color=BASE_C, ls=":", lw=1.8, zorder=2)
    which = "unseen SAE (131k features)" if a.sae == "unseen" else "trained SAE, held-out features"
    top = max(100 * p["evals"][k] for k, _, _ in SAE for _, p in sft_pts + rl_pts)
    if a.strength:
        ax_b.set_title(f"(b) {which}: strength"); ax_b.set_ylabel("target activation (% of corpus peak)")
    else:
        ax_b.set_title(f"(b) {which}: fires"); ax_b.set_ylabel("% of SAE features fired")
    step = 10 if top > 40 else 5
    ax_b.set_ylim(0, step * (int(top // step) + 1)); ax_b.yaxis.set_major_locator(plt.MultipleLocator(step))
    if a.sae == "trained":
        ax_b.legend(handles=[Line2D([], [], color="#444444", lw=1.8, ls=ls, label=lab) for _, ls, lab in SAE], loc="upper left", frameon=False)
    xs = [x for x, _ in sft_pts + rl_pts]; lo, hi = min(xs) / 1.3, max(xs) * 1.3
    ts = [t for t in cve.doubling_ticks(min(xs), hi) if t >= lo]
    for ax in (ax_a, ax_b):
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_xticks(ts, [f"{t:g}" for t in ts])
        ax.set_xlabel(r"training compute ($10^{19}$ FLOPs)"); ax.grid(True); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.0)
    os.makedirs(f"{cve.OUT}/data", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{cve.OUT}/{stem}.{ext}", dpi=150)
    data["sae_metric"] = metric
    json.dump(data, open(f"{cve.OUT}/data/{stem}.json", "w"), indent=1)
    print("wrote", f"{cve.OUT}/{stem}.png + .pdf + data/{stem}.json")


if __name__ == "__main__":
    main()
