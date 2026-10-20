#!/usr/bin/env python3
"""Per-landmark gain of PACER (4-stage) over its base: Tab. supp_perlm (Suppl.).

Every PACER run is paired with the base detector it was finetuned from (same
seed, labeled split and initialization). For each landmark k the per-landmark
test MRE is averaged over the seeds, giving one base value b_k and one PACER
value p_k. Landmarks are sorted by b_k and cut into three groups of
floor(K/3) landmarks (remainder to the middle group); each group column reads
mean b_k -> mean p_k. ``improved`` counts p_k < b_k, and r is the Pearson
correlation across landmarks between b_k and the absolute reduction b_k - p_k.

Parses the ``--- Per Landmark ---`` block of the ``test_metrics.txt`` reports
already written by train-base / finetune. No GPU, no retraining.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BUDGETS, DATASETS, SEEDS, test_metrics_txt  # noqa: E402


def per_landmark_mre(path):
    import numpy as np
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        blocks = fh.read().split("--- Per Landmark ---")
    if len(blocks) < 2:
        return None
    out = []
    for line in blocks[1].strip().splitlines()[1:]:  # skip the header row
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].startswith("L"):
            out.append(float(parts[1]))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-landmark PACER-vs-base gain table")
    ap.add_argument("--results", required=True, help="results root (output_dir of the runs)")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--method", default="pacer4", help="finetuned run compared with its base")
    args = ap.parse_args()

    import numpy as np

    for budget in BUDGETS:
        print(f"\n================ {budget}-shot ================")
        rows = []
        for ds, title, unit in DATASETS:
            base, ft = [], []
            for s in args.seeds:
                b = per_landmark_mre(test_metrics_txt(args.results, ds, s, budget, "base"))
                p = per_landmark_mre(test_metrics_txt(args.results, ds, s, budget, args.method))
                if b is not None and p is not None:
                    base.append(b)
                    ft.append(p)
            if not base:
                print(f"### {title}: missing runs")
                continue
            mb, mp = np.mean(base, 0), np.mean(ft, 0)  # per-landmark MRE averaged over seeds
            d = mp - mb
            improved = int((d < 0).sum())
            q = np.argsort(mb)                       # rank landmarks by BASE difficulty
            n3 = max(1, len(q) // 3)
            groups = [("easiest third", q[:n3]), ("middle third", q[n3:len(q) - n3]),
                      ("hardest third", q[-n3:])]
            r = np.corrcoef(mb, -d)[0, 1]
            print(f"\n### {title} ({unit})  K={len(mb)}  seeds={len(base)}  "
                  f"group sizes = {'/'.join(str(len(g)) for _, g in groups)}")
            print(f"  overall        {mb.mean():6.2f} -> {mp.mean():6.2f}   improved {improved}/{len(mb)}")
            for name, g in groups:
                print(f"  {name:14s} {mb[g].mean():6.2f} -> {mp[g].mean():6.2f} "
                      f"({100 * (mp[g].mean() - mb[g].mean()) / mb[g].mean():+.0f}%)")
            print(f"  Pearson r(base error, absolute reduction) = {r:+.2f}")
            o = np.argsort(d)
            print("  largest reductions: " + ", ".join(f"L{i + 1:02d} {mb[i]:.2f}->{mp[i]:.2f}" for i in o[:3]))
            print("  smallest/negative:  " + ", ".join(f"L{i + 1:02d} {mb[i]:.2f}->{mp[i]:.2f}" for i in o[-3:]))
            rows.append((title, improved, len(mb), [(mb[g].mean(), mp[g].mean()) for _, g in groups], r))
        print(f"\n--- LaTeX rows ({budget}-shot) ---")
        for title, imp, K, grp, r in rows:
            cells = " & ".join(f"${a:.2f}\\!\\to\\!{b:.2f}$" for a, b in grp)
            print(f"% {title}")
            print(f"${budget}$-shot & ${imp}/{K}$ & {cells} & ${r:+.2f}$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
