#!/usr/bin/env python3
"""Shape of the admission ramp: Tabs. supp_shape5/10 and Fig. shape (Suppl.).

Compares three ramps pi_t = Pi * (t/T)^gamma at matched T=4, Pi=100, E=120 and
matched base checkpoints: concave (gamma=1/2), linear (gamma=1, the reported
PACER) and convex (gamma=2). The runs are produced, for every dataset, seed and
budget, by

  python main.py finetune --config pacer/ablation/ramp_concave.json --override ...
  python main.py finetune --config pacer/configs/finetune.json      --override ...
  python main.py finetune --config pacer/ablation/ramp_convex.json  --override ...

This script prints the per-arm statistics, writes ``table_shape.tex`` (both
budgets; best in bold, second underlined, per column within a dataset) and
``schedule_shape.pdf`` (panel (a) the ramps, (b)-(d) the 5-shot test MRE per
dataset, mean +- std over seeds).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BUDGETS, DATASETS, SEEDS, mean_std, pm, test_metrics  # noqa: E402

T = 4
# (key, gamma, method key in common.RUN_DIRS, table label, colour); fixed order.
ARMS = [
    ("concave", 0.5, "pacer4_concave", r"PACER ($4$-stage, concave)", "#d95f02"),
    ("linear", 1.0, "pacer4", r"PACER ($4$-stage, linear)", "#1b9e77"),
    ("convex", 2.0, "pacer4_convex", r"PACER ($4$-stage, convex)", "#7570b3"),
]
CAPTIONS = {
    5: (r"Sensitivity to the \emph{shape} of the admission ramp at $5$ shots "
        r"($T{=}4$, $\Pi{=}100$, matched budget and base). "
        r"$\pi_t=\Pi(t/T)^{\gamma}$ with $\gamma=\tfrac12$ (concave), "
        r"$1$ (linear, the reported PACER) and $2$ (convex). Best per column "
        r"within a dataset in \textbf{bold}, second best \underline{underlined}."),
    10: (r"Sensitivity to the \emph{shape} of the admission ramp at $10$ shots; "
         r"settings and conventions as in Tab.~\ref{tab:supp_shape5}."),
}
SHORT = {"chest": "Shenzhen (chest)", "cephalo": "ISBI2015 (cephalo)", "hand": "DHA (hand)"}


def collect(results, seeds):
    stats = {}
    for ds, _, _ in DATASETS:
        for budget in BUDGETS:
            for key, _, method, _, _ in ARMS:
                got = [v for v in (test_metrics(results, ds, s, budget, method) for s in seeds)
                       if v is not None]
                if got:
                    stats[(ds, budget, key)] = (*mean_std(got), len(got))
    return stats


def write_table(stats, path):
    out = []
    for budget in BUDGETS:
        lines = [r"\begin{table}[tp]", r"\centering",
                 r"\caption{" + CAPTIONS[budget] + "}",
                 r"\label{tab:supp_shape" + str(budget) + "}",
                 r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lccc}", r"\toprule",
                 r"\textbf{Ramp} & \textbf{MRE}$\downarrow$ & "
                 r"\textbf{SDR@2}$\uparrow$ & \textbf{P95}$\downarrow$ \\", r"\midrule"]
        for di, (ds, title, unit) in enumerate(DATASETS):
            if di:
                lines.append(r"\midrule")
            unit_txt = "pixels" if unit == "px" else "mm"
            lines.append(r"\multicolumn{4}{l}{\emph{" + f"{title}, {unit_txt}" + r"}}\\")
            have = [k for k, *_ in ARMS if (ds, budget, k) in stats]
            # SDR@2: larger is better; MRE and P95: smaller is better.
            order = {c: sorted(have, key=lambda k: stats[(ds, budget, k)][0][c], reverse=(c == 1))
                     for c in range(3)}
            for key, _, _, label, _ in ARMS:
                if key not in have:
                    lines.append(f"{label} & " + " & ".join(["--"] * 3) + r" \\")
                    continue
                mu, sd, _ = stats[(ds, budget, key)]
                cells = []
                for c in range(3):
                    txt = pm(mu[c], sd[c])
                    if len(order[c]) > 1 and order[c][0] == key:
                        txt = r"\textbf{" + txt + "}"
                    elif len(order[c]) > 2 and order[c][1] == key:
                        txt = r"\underline{" + txt + "}"
                    cells.append(txt)
                lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        out.append("\n".join(lines))
    with open(path, "w") as fh:
        fh.write("\n\n".join(out) + "\n")


def plot(stats, path, budget=5):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn at the true double-column size so LaTeX does not rescale the fonts.
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.62))
    ax = axs[0]
    stages = np.arange(1, T + 1)
    labels = {"concave": r"concave $\gamma=\frac{1}{2}$", "linear": r"linear $\gamma=1$",
              "convex": r"convex $\gamma=2$"}
    for key, g, _, _, c in ARMS:
        ax.plot(stages, 100.0 * (stages / T) ** g, marker="o", ms=3.0, lw=1.3, color=c,
                label=labels[key], ls="--" if key == "linear" else "-")
    ax.set_title("(a) admission ramp", fontsize=7, pad=3)
    ax.set_xlabel("curriculum stage $t$", fontsize=7, labelpad=1.5)
    ax.set_ylabel(r"gate level $\pi_t$ (%)", fontsize=7, labelpad=1.5)
    ax.set_xticks(stages)
    ax.set_ylim(0, 124)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.3,
              borderaxespad=0.15, labelspacing=0.22, handletextpad=0.4)

    x = np.arange(len(ARMS))
    keys = [k for k, *_ in ARMS]
    for tag, ax, (ds, _, unit) in zip("bcd", axs[1:], DATASETS):
        if not all((ds, budget, k) in stats for k in keys):
            ax.set_title(f"({tag}) {SHORT[ds]}: missing runs", fontsize=7)
            continue
        mu = np.array([stats[(ds, budget, k)][0][0] for k in keys])
        sd = np.array([stats[(ds, budget, k)][1][0] for k in keys])
        ax.bar(x, mu, width=0.56, color=[c for *_, c in ARMS], yerr=sd, capsize=1.8,
               error_kw=dict(lw=0.7, ecolor="#555555"))
        for xi, m, s in zip(x, mu, sd):
            ax.text(xi, m + s + 0.06 * mu.max(), f"{m:.2f}", ha="center", fontsize=5.6,
                    color="#333333")
        ax.set_title(f"({tag}) {SHORT[ds]}, {unit}", fontsize=7, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["conc.", "lin.", "conv."], fontsize=6)
        ax.set_ylim(0, (mu + sd).max() * 1.32)
        ax.set_xlim(-0.62, len(ARMS) - 0.38)
    axs[1].set_ylabel("test MRE", fontsize=7, labelpad=1.5)
    for ax in axs:
        ax.tick_params(labelsize=6, pad=1.5, length=2)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    plt.tight_layout(pad=0.25, w_pad=0.9)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Ramp-shape ablation: LaTeX table + figure")
    ap.add_argument("--results", required=True, help="results root (output_dir of the runs)")
    ap.add_argument("--out", default=".", help="where to write table_shape.tex / schedule_shape.pdf")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    stats = collect(args.results, args.seeds)
    print(f"{'dataset':9s} {'bud':>3s} {'arm':8s} {'n':>2s}  {'MRE':>13s} {'SDR@2':>13s} {'P95':>13s}")
    for ds, _, _ in DATASETS:
        for budget in BUDGETS:
            for key, *_ in ARMS:
                if (ds, budget, key) not in stats:
                    print(f"{ds:9s} {budget:3d} {key:8s}  missing")
                    continue
                mu, sd, n = stats[(ds, budget, key)]
                print(f"{ds:9s} {budget:3d} {key:8s} {n:2d}  "
                      + "  ".join(f"{mu[i]:6.2f}+-{sd[i]:5.2f}" for i in range(3)))

    os.makedirs(args.out, exist_ok=True)
    write_table(stats, os.path.join(args.out, "table_shape.tex"))
    plot(stats, os.path.join(args.out, "schedule_shape.pdf"))
    print(f"wrote {os.path.join(args.out, 'table_shape.tex')} and schedule_shape.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
