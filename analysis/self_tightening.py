"""Suppl. Fig. "fig:self_tightening" (self_tightening_caps.pdf).

Mean ERE of the admitted set at the entry of each stage of the 4-stage PACER
curriculum, plotted against the admitted fraction, for the default cap
(Pi=100, run ``pacer4``) and the tail-withholding cap (Pi=60, run
``pacer4_cap60``). Rows: 5-shot, 10-shot; columns: datasets; mean +/- std over
SEEDS. Reads only ``pseudo_refresh_log.json`` of each run.

Usage: python analysis/self_tightening.py --results <results> --out <figdir>
"""

import argparse
import json
import os

# (run key, label, color, marker, linestyle)
CAPS = [("pacer4", r"$\Pi=100$", "#2c6fbb", "o", "-"),
        ("pacer4_cap60", r"$\Pi=60$", "#d1495b", "s", "--")]


def collect(results, ds, method, budget, seeds):
    import numpy as np
    from common import run_dir
    rows, pcts = [], None
    for s in seeds:
        with open(os.path.join(run_dir(results, ds, s, budget, method), "pseudo_refresh_log.json")) as fh:
            log = json.load(fh)
        # (second key: logs written by the pre-release research code)
        rows.append([e.get("mean_selected_sample_ere", e.get("mean_selected_sample_mean_ere")) for e in log])
        pcts = [e["percentile"] for e in log]
    a = np.array(rows, dtype=np.float64)
    # population std (ddof=0), as in the paper figure
    return np.array(pcts), a.mean(0), a.std(0)


def main():
    from common import BUDGETS, DATASETS, SEEDS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="root of the run tree")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    handles = labels = None
    for ri, budget in enumerate(BUDGETS):
        for ci, (ds, title, unit) in enumerate(DATASETS):
            ax = axes[ri, ci]
            for method, lab, col, mk, ls in CAPS:
                pcts, mean, std = collect(args.results, ds, method, budget, args.seeds)
                ax.errorbar(pcts, mean, yerr=std, marker=mk, markersize=9,
                            capsize=5, capthick=1.8, elinewidth=1.8, linewidth=3.2,
                            linestyle=ls, color=col, label=lab, zorder=3)
            ax.set_xticks([20, 40, 60, 80, 100])
            ax.set_xlim(8, 108)
            ax.margins(y=0.30)
            ax.grid(True, axis="y", linestyle=":", linewidth=1.0, color="0.8", zorder=0)
            ax.tick_params(labelsize=18, length=5, width=1.2)
            for sp in ax.spines.values():
                sp.set_linewidth(1.3)
            ax.spines[["top", "right"]].set_visible(False)
            if ri == 0:
                ax.set_title(f"{title}, {unit}", fontsize=18, pad=8)
            if ri == 1:
                ax.set_xlabel(r"admitted fraction $\pi_t$ (%)", fontsize=20)
            if ci == 0:
                ax.set_ylabel("mean admitted ERE", fontsize=20)
                ax.text(-0.46, 0.5, f"${budget}$-shot", transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontweight="bold", fontsize=26)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=23, bbox_to_anchor=(0.5, 1.03), columnspacing=2.0, handletextpad=0.6)
    fig.tight_layout(pad=0.6, w_pad=2.5, rect=(0.02, 0, 1, 0.95))
    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"self_tightening_caps.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(args.out, "self_tightening_caps.pdf"))


if __name__ == "__main__":
    main()
