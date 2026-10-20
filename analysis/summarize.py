#!/usr/bin/env python3
"""Mean +- std tables over seeds for any set of runs (paper tables).

Reads ``metrics_final.json`` / ``base_metrics.json`` of every requested
(dataset, budget, seed, run) and prints test MRE, SDR@2 and P95 as
mean +- sample std (ddof=1) over the seeds found, i.e. the cell format of
every results table in the paper. ``--ref`` adds a paired t-test on MRE
against a reference run over the seeds both runs share.

Runs are method keys of ``analysis/common.py`` (base, pacer4, pacer4_cap60,
pacer4_concave, pacer4_convex, mean_teacher, fixmatch_pose, flexmatch,
freematch) or any raw run sub-directory, e.g. ``step-8_cap100/random`` or
``ssl/stld_hrnet``.

Examples:
  # Tab. main5, U-Net group (use --budgets 10 for Tab. main10)
  python analysis/summarize.py --results R --budgets 5 \\
      --runs base mean_teacher fixmatch_pose flexmatch freematch pacer4 --ref pacer4
  # Tabs. supp_staging5/10
  python analysis/summarize.py --results R --seeds 14 42 90 --runs \\
      flat_cap100/percentile step-2_cap100/percentile step-4_cap100/percentile step-8_cap100/percentile
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ALL_SEEDS, BUDGETS, DATASETS, mean_std, test_metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Mean +- std over seeds of test MRE / SDR@2 / P95",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--results", required=True, help="results root (output_dir of the runs)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=[d for d, _, _ in DATASETS])
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=ALL_SEEDS)
    ap.add_argument("--ref", default=None, help="reference run for a paired t-test on MRE")
    args = ap.parse_args()

    from scipy.stats import ttest_rel

    runs = list(dict.fromkeys(args.runs + ([args.ref] if args.ref else [])))
    for budget in args.budgets:
        print(f"\n===== {budget}-shot =====")
        for ds in args.datasets:
            print(f"--- {ds} ---")
            per_run = {r: {s: test_metrics(args.results, ds, s, budget, r) for s in args.seeds}
                       for r in runs}
            for r in args.runs:
                got = {s: v for s, v in per_run[r].items() if v is not None}
                if not got:
                    print(f"  {r:32s} (no runs)")
                    continue
                mu, sd = mean_std(list(got.values()))
                line = (f"  {r:32s} n={len(got)}  MRE {mu[0]:7.2f} +- {sd[0]:5.2f}   "
                        f"SDR@2 {mu[1]:6.2f} +- {sd[1]:5.2f}   P95 {mu[2]:7.2f} +- {sd[2]:6.2f}")
                if args.ref and r != args.ref:
                    shared = [s for s in got if per_run[args.ref].get(s) is not None]
                    if len(shared) >= 2:
                        p = ttest_rel([got[s][0] for s in shared],
                                      [per_run[args.ref][s][0] for s in shared]).pvalue
                        line += f"   paired-t vs {args.ref} (n={len(shared)}): p={p:.3f}"
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
