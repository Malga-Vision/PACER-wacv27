"""Ranking quality of the label-free uncertainty signals on the base model.

Produces Fig. ranking (``ranking_quality_5shot.pdf``), Fig. ranking10
(``ranking_quality_10shot.pdf``) and the LaTeX rows of Tab. corr.

For every dataset, budget {5, 10} and base-model seed (default: all five), the
unlabeled residual pool is scored with that seed's BASE checkpoint. Per sample
(mean over landmarks):
  ere   Expected Radial Error under the heatmap (PACER's signal)
  conf  peak (mode) softmax probability
  ent   heatmap Shannon entropy, -sum p log p
  tta   11-view TTA disagreement (analysis/tta.py; --skip-tta to omit)
  gtr   anisotropic-Gaussian read-out sqrt(trace Sigma) of the heatmap,
        on the same >5%-of-peak support ERE uses
  mre   true radial error (ground truth revealed, analysis only)

Outputs: Spearman rho / Pearson r of each signal vs. true MRE (mean over seeds),
and the admitted-subset curves: true MRE of the top-k% most-trusted samples
(ERE/entropy/TTA/gtr ascending, confidence descending), mean +/- std over seeds.
Raw per-run arrays are cached in ``<out>/ranking_data.npz``; ``--plot-only``
re-plots from the cache without a GPU. Re-runs skip cells already cached.

  python analysis/ranking_quality.py --results <results> --dataset-root <datasets> --out <out>
"""

import argparse
import json
import os

from common import ALL_SEEDS, BUDGETS, DATASETS, checkpoint_path, eval_cfg, run_dir

CUT = 0.05  # significant-pixel cutoff of shared.metrics.calculate_ere
# (key, legend label, most-trusted-first sort sign (+1 low=good, -1 high=good), colour)
METRICS = [("ere", "ERE", +1, "#d62728"),
           ("conf", "Peak conf.", -1, "#1f77b4"),
           ("ent", "Entropy", +1, "#2ca02c"),
           ("tta", "TTA disag.", +1, "#9467bd"),
           ("gtr", "Gauss. cov.", +1, "#ff7f0e")]
# panel / table order of the paper figure
DS_ORDER = ["hand", "cephalo", "chest"]
SHORT = {"hand": "DHA", "cephalo": "ISBI", "chest": "Shen."}
UNIT = {ds: unit for ds, _, unit in DATASETS}


def fracs():
    import numpy as np
    return np.arange(10, 101, 10)  # admission percentages


def gaussian_trace(heatmap, pixel_size, cut=CUT):
    """sqrt(trace Sigma) of a moment-fitted anisotropic Gaussian, in physical units."""
    import numpy as np
    w = np.asarray(heatmap, dtype=np.float64)
    peak = w.max()
    if peak <= 0:
        return 0.0
    w = np.where(w / peak > cut, w, 0.0)
    tot = w.sum()
    if tot <= 1e-12:
        return 0.0
    p = (w / tot).ravel()
    ys, xs = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    x = xs.ravel() * float(pixel_size[0])
    y = ys.ravel() * float(pixel_size[1])
    dx, dy = x - (p * x).sum(), y - (p * y).sum()
    return float(np.sqrt(max(float((p * dx * dx).sum()) + float((p * dy * dy).sum()), 0.0)))


def score_pool(cfg, model, records, device):
    """Per-sample ere, conf, ent, gtr, mre arrays in the order of ``records``."""
    import numpy as np
    import torch
    from shared.data import create_dataloader
    from shared.metrics import CustomNLLLoss, summarize_sample_errors

    loader = create_dataloader(records=records, cfg=cfg, shuffle=False, seed=14)
    acc = {k: {} for k in ("ere", "conf", "ent", "gtr", "mre")}
    with torch.no_grad():
        for batch in loader:
            probs = CustomNLLLoss.two_d_softmax(model(batch["image"].to(device))).cpu().numpy()
            scal = batch["physical_scaling_factor"].cpu().numpy()
            gts = batch["landmarks"].cpu().numpy()
            for i, sid in enumerate(batch["name"]):
                sid = str(sid)
                s = summarize_sample_errors(probs[i], gts[i], scal[i])
                p = probs[i].reshape(probs[i].shape[0], -1)
                acc["ere"][sid] = float(np.mean(s["per_landmark_ere"]))
                acc["mre"][sid] = float(np.mean(s["per_landmark_mre"]))
                acc["conf"][sid] = float(np.mean(s["mode_probs"]))
                acc["ent"][sid] = float(np.mean(-(p * np.log(p + 1e-12)).sum(axis=1)))
                acc["gtr"][sid] = float(np.mean([gaussian_trace(probs[i][k], scal[i])
                                                 for k in range(probs[i].shape[0])]))
    ids = [str(r["sample_id"]) for r in records]
    return {k: np.array([acc[k][s] for s in ids]) for k in acc}


def collect(args, cache):
    """Score every (dataset, budget, seed) base run not yet in ``cache``."""
    import numpy as np
    import torch
    from shared.data import load_dataset_splits
    from shared.model import create_landmark_model

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    need = {"ere", "conf", "ent", "gtr", "mre"} | (set() if args.skip_tta else {"tta"})
    for budget in BUDGETS:
        for ds in DS_ORDER:
            for seed in args.seeds:
                key = (ds, budget, seed)
                if need <= set(cache.get(key, {})):
                    continue
                base = run_dir(args.results, ds, seed, budget, "base")
                cfg = eval_cfg(args.results, ds, seed, budget, args.dataset_root)
                splits, K = load_dataset_splits(cfg)
                by_id = {r["sample_id"]: r for r in splits["train"]}
                with open(os.path.join(base, "residual_ids.json")) as fh:
                    pool = [s for s in json.load(fh) if s in by_id]
                records = [by_id[s] for s in pool]
                model = create_landmark_model(cfg, K, device)
                ck = torch.load(checkpoint_path(args.results, ds, seed, budget, "base"), map_location=device, weights_only=True)
                model.load_state_dict(ck.get("model_state_dict", ck))
                model.eval()
                v = score_pool(cfg, model, records, device)
                if not args.skip_tta:
                    from tta import compute_tta_disagreement
                    tab = compute_tta_disagreement(model=model, records=records, cfg=cfg, device=device)
                    v["tta"] = np.array([float(tab[s]["sample_mean_mm"]) for s in pool])
                cache[key] = v
                print(f"[{budget}-shot {ds} seed {seed}] n={len(pool)}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                save_cache(args.cache, cache)  # checkpoint after every run
    return cache


def save_cache(path, cache):
    """Flat .npz: one array per '<ds>|<budget>|<seed>|<signal>' key."""
    import numpy as np
    np.savez(path, **{f"{ds}|{b}|{s}|{k}": a for (ds, b, s), v in cache.items() for k, a in v.items()})


def load_cache(path):
    import numpy as np
    cache = {}
    if os.path.exists(path):
        with np.load(path) as z:
            for name in z.files:
                ds, b, s, k = name.split("|")
                cache.setdefault((ds, int(b), int(s)), {})[k] = z[name]
    return cache


def admitted_curve(vals, mre, sign):
    """True MRE of the top-k% admitted subset, most-trusted first."""
    import numpy as np
    order = np.argsort(sign * vals)
    n = len(mre)
    return np.array([mre[order[:max(1, int(np.ceil(f / 100 * n)))]].mean() for f in fracs()])


def summarize(cache, seeds, metrics):
    """curves[budget][ds][mk] -> list over seeds; rhos/pears[(mk, ds, budget)] -> list."""
    from scipy.stats import pearsonr, spearmanr
    curves = {b: {ds: {mk: [] for mk, *_ in metrics} for ds in DS_ORDER} for b in BUDGETS}
    rhos, pears = {}, {}
    for b in BUDGETS:
        for ds in DS_ORDER:
            for seed in seeds:
                v = cache.get((ds, b, seed))
                if v is None:
                    print(f"warning: missing {ds} {b}-shot seed {seed}")
                    continue
                for mk, _, sign, _ in metrics:
                    rhos.setdefault((mk, ds, b), []).append(spearmanr(v[mk], v["mre"])[0])
                    pears.setdefault((mk, ds, b), []).append(pearsonr(v[mk], v["mre"])[0])
                    curves[b][ds][mk].append(admitted_curve(v[mk], v["mre"], sign))
    return curves, rhos, pears


def plot_ranking(curves, metrics, budget, fname):
    """Single-column (~\\columnwidth) figure so fonts stay ~7pt in print."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    F = fracs()
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.55))
    for ax, ds in zip(axs, DS_ORDER):
        for mk, label, _, color in metrics:
            if not curves[budget][ds][mk]:
                continue
            arr = np.stack(curves[budget][ds][mk])  # (seeds, fracs)
            mu, sd = arr.mean(0), arr.std(0)
            ax.plot(F, mu, marker="o", ms=2.4, lw=1.1, color=color, label=label)
            ax.fill_between(F, mu - sd, mu + sd, color=color, alpha=0.12, lw=0)
        ax.set_title(f"{SHORT[ds]} ({UNIT[ds]})", fontsize=7, pad=2)
        ax.set_xticks([25, 50, 75, 100])
        ax.tick_params(labelsize=6, pad=1.5, length=2)
        ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("subset MRE", fontsize=7)
    fig.supxlabel("admitted top-$k$ (%)", fontsize=7, y=0.04)
    axs[0].legend(fontsize=5, frameon=False, loc="upper left", handlelength=1.0,
                  borderaxespad=0.15, labelspacing=0.2, handletextpad=0.4)
    plt.tight_layout(pad=0.2, w_pad=0.5)
    plt.subplots_adjust(bottom=0.24)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print("wrote", fname)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="results root (run tree of the training entrypoints)")
    ap.add_argument("--dataset-root", help="directory holding the landmark datasets")
    ap.add_argument("--out", required=True, help="output dir for the cache, PDFs")
    ap.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    ap.add_argument("--skip-tta", action="store_true", help="omit the (expensive) TTA signal")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from <out>/ranking_data.npz")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    args = ap.parse_args()
    if not args.plot_only and not (args.results and args.dataset_root):
        ap.error("--results and --dataset-root are required unless --plot-only")

    import numpy as np
    os.makedirs(args.out, exist_ok=True)
    args.cache = os.path.join(args.out, "ranking_data.npz")
    cache = load_cache(args.cache)
    if not args.plot_only:
        cache = collect(args, cache)

    have_tta = not args.skip_tta and all("tta" in cache.get((ds, b, s), {"tta": 0})
                                         for ds in DS_ORDER for b in BUDGETS for s in args.seeds)
    metrics = [m for m in METRICS if m[0] != "tta" or have_tta]
    curves, rhos, pears = summarize(cache, args.seeds, metrics)

    for b in BUDGETS:
        plot_ranking(curves, metrics, b, os.path.join(args.out, f"ranking_quality_{b}shot.pdf"))

    print("\n% ---- correlation vs. true error (mean +/- std over seeds) ----")
    for b in BUDGETS:
        for mk, label, *_ in metrics:
            for ds in DS_ORDER:
                r, p = rhos.get((mk, ds, b), [np.nan]), pears.get((mk, ds, b), [np.nan])
                print(f"%  {b}sh {label:11s} {ds:7s} rho={np.mean(r):+.3f}+/-{np.std(r):.3f}"
                      f"  r={np.mean(p):+.3f}+/-{np.std(p):.3f}")

    # Tab. corr rows: Signal & rho & r per dataset (DHA, ISBI, Shenzhen), mean over seeds
    print("\n% ---- correlation rows (mean over seeds) ----")
    for b in BUDGETS:
        print(f"% {b}-shot")
        for mk, label, *_ in metrics:
            row = []
            for ds in DS_ORDER:
                row += [f"${np.mean(rhos.get((mk, ds, b), [np.nan])):+.2f}$",
                        f"${np.mean(pears.get((mk, ds, b), [np.nan])):+.2f}$"]
            print(f"{label} & " + " & ".join(row) + r" \\")


if __name__ == "__main__":
    main()
