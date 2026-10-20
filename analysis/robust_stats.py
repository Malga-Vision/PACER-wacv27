#!/usr/bin/env python3
"""Robust error statistics: Tabs. supp_robust5/10 (Suppl.).

Median and interquartile width of the per-landmark test radial errors pooled
over images, next to MRE and P95, for the base detector, the four SSL baselines
and PACER (4-stage), at both budgets: mean +- std over the seeds of each cell.
MRE and P95 are read from the run metrics (the source of the main tables);
median and IQR width come from the dumps of ``dump_sample_errors.py``.

Each cell is averaged over the seeds its main-table cell uses
(``--seed-basis``, default ``analysis/seed_basis.json``); pass
``--seed-basis none`` to use every seed found. Emits the LaTeX table body
(best per column within a sub-block in bold, second underlined) and the
base -> PACER median / IQR changes quoted in the text.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ALL_SEEDS, BUDGETS, DATASETS, mean_std, pm, test_metrics  # noqa: E402

METHODS = [("base", "ResNet-101 (ImageNet)"),
           ("mean_teacher", "Mean Teacher"),
           ("fixmatch_pose", "FixMatch-pose"),
           ("flexmatch", "FlexMatch"),
           ("freematch", "FreeMatch"),
           ("pacer4", "PACER (4-stage)")]


def cell_stats(args, basis, ds, budget, method):
    """(mean, std) over the cell's seeds of (MRE, median, IQR width, P95), or None."""
    import numpy as np
    seeds = basis.get(f"{ds}_{budget}_{method}", ALL_SEEDS) if basis else ALL_SEEDS
    rows = []
    for s in seeds:
        dump = os.path.join(args.errors, f"{ds}_{budget}_{s}_{method}.json")
        metrics = test_metrics(args.results, ds, s, budget, method)
        if not os.path.exists(dump) or metrics is None:
            if basis:
                return None  # a cell must rest on exactly its main-table seeds
            continue
        with open(dump) as fh:
            e = np.concatenate([np.asarray(v) for v in json.load(fh).values()])
        rows.append([metrics[0], np.median(e), np.percentile(e, 75) - np.percentile(e, 25), metrics[2]])
    return mean_std(rows) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Median / IQR robust statistics table")
    ap.add_argument("--results", required=True, help="results root (output_dir of the runs)")
    ap.add_argument("--errors", required=True, help="directory written by dump_sample_errors.py")
    ap.add_argument("--seed-basis", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "seed_basis.json"))
    args = ap.parse_args()

    basis = {}
    if args.seed_basis.lower() != "none":
        with open(args.seed_basis) as fh:
            basis = json.load(fh)

    tex, table = [], {}
    for di, (ds, title, unit) in enumerate(DATASETS):
        if di:
            tex.append("\\midrule")
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}, {unit}}}}}\\\\")
        for b in BUDGETS:
            tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{${b}$-shot}}}}\\\\")
            for m, _ in METHODS:
                table[(ds, b, m)] = cell_stats(args, basis, ds, b, m)
            have = [m for m, _ in METHODS if table[(ds, b, m)] is not None]
            rank = {c: sorted(have, key=lambda m: table[(ds, b, m)][0][c]) for c in range(4)}
            for m, label in METHODS:
                st = table[(ds, b, m)]
                if st is None:
                    tex.append(f"{label} & " + " & ".join(["--"] * 4) + " \\\\")
                    continue
                cells = []
                for c in range(4):
                    t = pm(st[0][c], st[1][c])
                    pos = rank[c].index(m)
                    cells.append(f"\\textbf{{{t}}}" if pos == 0 else f"\\underline{{{t}}}" if pos == 1 else t)
                lab = f"\\textbf{{{label}}}" if m == "pacer4" else label
                tex.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    print("\n".join(tex))

    print("\n% base -> PACER (4-stage): median and IQR width")
    for ds, _, _ in DATASETS:
        for b in BUDGETS:
            sb, sp = table[(ds, b, "base")], table[(ds, b, "pacer4")]
            if sb is None or sp is None:
                continue
            dmed = 100 * (sp[0][1] - sb[0][1]) / sb[0][1]
            diqr = 100 * (sp[0][2] - sb[0][2]) / sb[0][2]
            print(f"% {ds:8s} {b:2d}-shot  median {sb[0][1]:.2f} -> {sp[0][1]:.2f} ({dmed:+.0f}%)   "
                  f"IQR {sb[0][2]:.2f} -> {sp[0][2]:.2f} ({diqr:+.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
