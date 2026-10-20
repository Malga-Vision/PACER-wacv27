#!/usr/bin/env python3
"""Per-(sample, landmark) test radial errors from saved checkpoints.

Inference only. For every (dataset, budget, seed, method) found under
``--results`` it re-evaluates the checkpoint on the test split and writes
``<out>/<dataset>_<budget>_<seed>_<method>.json`` = {sample_id: [error per
landmark]} in the dataset's unit. These dumps feed ``robust_stats.py``
(median and IQR, Tabs. supp_robust5/10). Existing dumps are skipped.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ALL_SEEDS, BUDGETS, DATASETS, checkpoint_path, eval_cfg  # noqa: E402

METHODS = ["base", "pacer4", "mean_teacher", "fixmatch_pose", "flexmatch", "freematch"]


def dump(cfg, ckpt, device):
    import torch
    from shared.data import create_dataloader, load_dataset_splits
    from shared.metrics import CustomNLLLoss, summarize_sample_errors
    from shared.model import create_landmark_model

    splits, num_landmarks = load_dataset_splits(cfg)
    model = create_landmark_model(cfg, num_landmarks, device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    loader = create_dataloader(records=splits["test"], cfg=cfg, shuffle=False, seed=cfg["seed"])
    errors = {}
    with torch.no_grad():
        for batch in loader:
            probs = CustomNLLLoss.two_d_softmax(model(batch["image"].to(device))).cpu().numpy()
            scaling = batch["physical_scaling_factor"].cpu().numpy()
            gts = batch["landmarks"].cpu().numpy()
            for i, sid in enumerate(batch["name"]):
                errors[str(sid)] = summarize_sample_errors(probs[i], gts[i], scaling[i])["per_landmark_mre"]
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump per-(sample, landmark) test errors")
    ap.add_argument("--results", required=True, help="results root (output_dir of the runs)")
    ap.add_argument("--dataset-root", required=True, help="datasets directory (dataset_root)")
    ap.add_argument("--out", required=True, help="directory for the JSON dumps")
    ap.add_argument("--datasets", nargs="+", default=[d for d, _, _ in DATASETS])
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=ALL_SEEDS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    for ds in args.datasets:
        for budget in args.budgets:
            for seed in args.seeds:
                for method in args.methods:
                    ckpt = checkpoint_path(args.results, ds, seed, budget, method)
                    dst = os.path.join(args.out, f"{ds}_{budget}_{seed}_{method}.json")
                    if os.path.exists(dst):
                        print("skip", os.path.basename(dst), flush=True)
                        continue
                    if not os.path.exists(ckpt):
                        print("missing", ckpt, flush=True)
                        continue
                    cfg = eval_cfg(args.results, ds, seed, budget, args.dataset_root)
                    errors = dump(cfg, ckpt, device)
                    with open(dst, "w") as fh:
                        json.dump(errors, fh)
                    flat = np.concatenate([np.asarray(v) for v in errors.values()])
                    print(f"{ds} b{budget} s{seed} {method:14s} n={flat.size} mean={flat.mean():.4f} "
                          f"median={np.median(flat):.2f} IQR=[{np.percentile(flat, 25):.2f},"
                          f"{np.percentile(flat, 75):.2f}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
