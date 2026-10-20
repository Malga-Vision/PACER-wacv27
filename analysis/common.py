"""Shared helpers for the paper analysis scripts (tables and figures).

Every script reads the run tree written by the training entrypoints,
``<results>/<dataset>/seed_<s>/active_<N>/<run>/``, and never writes into it.
``RUN_DIRS`` maps the method keys used throughout ``analysis/`` to the
``<run>`` sub-directory each entrypoint produces with its default settings.
"""

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATASETS = [
    ("chest", "Shenzhen (chest)", "px"),
    ("cephalo", "ISBI2015 (cephalometric)", "mm"),
    ("hand", "DHA (hand)", "mm"),
]
BUDGETS = [5, 10]
SEEDS = [14, 42, 90]            # the seeds every PACER cell is averaged over
ALL_SEEDS = [7, 14, 42, 90, 123]  # base / SSL references may use all five

RUN_DIRS = {
    "base": "base",
    "pacer4": os.path.join("step-4_cap100", "percentile"),
    "pacer4_cap60": os.path.join("step-4_cap60", "percentile"),
    "pacer4_concave": os.path.join("step-4_cap100_gamma0p5", "percentile"),
    "pacer4_convex": os.path.join("step-4_cap100_gamma2", "percentile"),
    "mean_teacher": os.path.join("ssl", "mean_teacher"),
    "fixmatch_pose": os.path.join("ssl", "fixmatch_pose"),
    "flexmatch": os.path.join("ssl", "flexmatch"),
    "freematch": os.path.join("ssl", "freematch"),
}

CHECKPOINTS = {"base": "base_checkpoint.pt"}  # every other run: final_checkpoint.pt


def run_dir(results: str, ds: str, seed: int, budget: int, method: str) -> str:
    sub = RUN_DIRS.get(method, method)
    return os.path.join(results, ds, f"seed_{seed}", f"active_{budget}", sub)


def checkpoint_path(results: str, ds: str, seed: int, budget: int, method: str) -> str:
    return os.path.join(run_dir(results, ds, seed, budget, method),
                        CHECKPOINTS.get(method, "final_checkpoint.pt"))


def test_metrics(results: str, ds: str, seed: int, budget: int,
                 method: str) -> Optional[Tuple[float, float, float]]:
    """(MRE, SDR@2, P95) on the test split, or None if the run is absent."""
    d = run_dir(results, ds, seed, budget, method)
    for name in ("metrics_final.json", "base_metrics.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as fh:
                t = json.load(fh)
            t = t.get("test", t)
            return float(t["MRE"]), float(t["SDR"]["2.0mm"]), float(t["P95"])
    return None


def test_metrics_txt(results: str, ds: str, seed: int, budget: int, method: str) -> str:
    """Path of the human-readable test report (holds the per-landmark block)."""
    d = run_dir(results, ds, seed, budget, method)
    return os.path.join(d, "test_metrics.txt" if method == "base" else os.path.join("eval_final", "test_metrics.txt"))


def eval_cfg(results: str, ds: str, seed: int, budget: int,
             dataset_root: str, batch_size: int = 4) -> Dict[str, Any]:
    """Inference config for a (dataset, seed, budget) cell, from its base run_config."""
    cfg: Dict[str, Any] = {}
    p = os.path.join(run_dir(results, ds, seed, budget, "base"), "run_config.json")
    if os.path.exists(p):
        with open(p) as fh:
            cfg = json.load(fh)
    cfg.update(dict(image_size=(256, 256), image_channels=1, sigma=0.0, num_workers=4,
                    pin_memory=True, batch_size=batch_size, architecture="smp_unet",
                    encoder="resnet101", encoder_weights="imagenet", pretrained_path=None,
                    dataset=ds, dataset_root=dataset_root, seed=seed, initial_labeled=budget))
    return cfg


def mean_std(rows):
    """Column-wise mean and sample std (ddof=1, as in the paper tables)."""
    import numpy as np
    a = np.asarray(rows, dtype=np.float64)
    sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1:])
    return a.mean(0), sd


def pm(mu: float, sd: float) -> str:
    return rf"{mu:.2f}\,$\pm$\,{sd:.2f}"
