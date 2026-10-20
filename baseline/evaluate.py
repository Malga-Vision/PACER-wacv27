"""Comprehensive evaluation for PACER landmark detectors."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data import create_dataloader
from shared.metrics import (
    CustomNLLLoss,
    compute_p95,
    compute_sdr,
    summarize_sample_errors,
)
from shared.visualization import plot_prediction_debug_grid, plot_split_overview


def strip_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "MRE": m["MRE"], "ERE": m["ERE"], "P95": m["P95"], "loss": m["loss"],
        "SDR": m["SDR"], "sample_ids": m["sample_ids"],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    scalar = float(value)
    if np.isnan(scalar) or np.isinf(scalar):
        return "NA"
    return f"{scalar:.{digits}f}"


def _sorted_sdr_pairs(sdr: Dict[str, Any]) -> List[Tuple[str, float]]:
    pairs = []
    for key, value in sdr.items():
        numeric = float(str(key).replace("mm", "").strip())
        pairs.append((numeric, str(key), float(value)))
    pairs.sort(key=lambda row: row[0])
    return [(label, score) for _, label, score in pairs]


def _write_split_metrics_txt(split_name: str, metrics: Dict[str, Any], output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    per_lm_mre = [float(v) for v in metrics.get("per_landmark_MRE", [])]
    per_lm_ere = [float(v) for v in metrics.get("per_landmark_ERE", [])]
    per_lm_p95 = [float(v) for v in metrics.get("per_landmark_P95", [])]

    lines = [
        f"===== {split_name.upper()} EVALUATION =====",
        "",
        "--- Global Metrics ---",
        f"Loss:    {_fmt(metrics.get('loss'), digits=6)}",
        f"MRE:     {_fmt(metrics.get('MRE'))} +/- {_fmt(metrics.get('MRE_std'))}",
        f"ERE:     {_fmt(metrics.get('ERE'))} +/- {_fmt(metrics.get('ERE_std'))}",
        f"P95:     {_fmt(metrics.get('P95'))}",
        f"Samples: {len(metrics.get('sample_ids', []))}",
        "",
        "--- SDR ---",
    ]
    for key, value in _sorted_sdr_pairs(metrics.get("SDR", {})):
        lines.append(f"  {key}: {_fmt(value, digits=2)}%")

    if per_lm_mre:
        lines += ["", "--- Per Landmark ---", "LM\tMRE\tERE\tP95"]
        for idx, mre in enumerate(per_lm_mre):
            ere = per_lm_ere[idx] if idx < len(per_lm_ere) else float("nan")
            p95 = per_lm_p95[idx] if idx < len(per_lm_p95) else float("nan")
            lines.append(f"L{idx + 1:02d}\t{_fmt(mre)}\t{_fmt(ere)}\t{_fmt(p95)}")

    with open(output_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    num_landmarks: int,
    seed: int,
    save_dir: Optional[str] = None,
    split_name: str = "test",
    grid_samples: int = 0,
) -> Dict[str, Any]:
    """Evaluate ``model`` on ``records`` and return a full metrics dictionary."""
    dataloader = create_dataloader(records=records, cfg=cfg, shuffle=False, seed=seed)
    loss_fn = CustomNLLLoss()

    model = model.to(device)
    model.eval()

    losses: List[float] = []
    radial_errors: List[float] = []
    expected_radial_errors: List[float] = []
    per_landmark_mre_values: List[List[float]] = [[] for _ in range(num_landmarks)]
    per_landmark_ere_values: List[List[float]] = [[] for _ in range(num_landmarks)]

    sample_ids: List[str] = []
    sample_ere_per_sample: List[float] = []
    sample_mre_per_sample: List[float] = []
    pred_landmarks_all: List[List[List[float]]] = []

    # For visualization
    vis_images: List[np.ndarray] = []
    vis_gt_landmarks: List[np.ndarray] = []
    vis_pred_landmarks: List[np.ndarray] = []
    vis_heatmaps: List[np.ndarray] = []
    vis_scaling: List[np.ndarray] = []
    vis_names: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            gt_landmarks = batch["landmarks"].cpu().numpy()
            scaling = batch["physical_scaling_factor"].cpu().numpy()
            names = [str(n) for n in batch["name"]]

            probs = CustomNLLLoss.two_d_softmax(model(images))
            losses.append(float(loss_fn(probs, targets).item()))

            probs_np = probs.cpu().numpy()

            # Collect for visualization
            if save_dir and grid_samples > 0 and len(vis_names) < grid_samples:
                vis_images.append(batch["image"].cpu().numpy())
                vis_heatmaps.append(probs_np)
                vis_gt_landmarks.append(gt_landmarks)
                vis_scaling.append(scaling)
                vis_names.extend(names)

            for i in range(probs_np.shape[0]):
                sample_ids.append(names[i])
                sample_summary = summarize_sample_errors(
                    heatmaps=probs_np[i],
                    gt_landmarks=gt_landmarks[i],
                    scale=scaling[i],
                )

                lm_preds = sample_summary["pred_landmarks"]
                lm_mres = sample_summary["per_landmark_mre"]
                lm_eres = sample_summary["per_landmark_ere"]
                img_pred_lm = np.asarray(lm_preds, dtype=np.float64)

                radial_errors.extend(lm_mres)
                expected_radial_errors.extend(lm_eres)
                for lm, value in enumerate(lm_mres):
                    per_landmark_mre_values[lm].append(float(value))
                for lm, value in enumerate(lm_eres):
                    per_landmark_ere_values[lm].append(float(value))

                pred_landmarks_all.append(lm_preds)
                sample_ere = float(np.mean(lm_eres))
                sample_mre = float(np.mean(lm_mres))
                sample_ere_per_sample.append(sample_ere)
                sample_mre_per_sample.append(sample_mre)

                if save_dir and grid_samples > 0 and len(vis_pred_landmarks) < grid_samples:
                    vis_pred_landmarks.append(img_pred_lm)

    re_arr = np.asarray(radial_errors, dtype=np.float64)
    ere_arr = np.asarray(expected_radial_errors, dtype=np.float64)

    results = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "MRE": float(np.mean(re_arr)) if re_arr.size else 0.0,
        "MRE_std": float(np.std(re_arr)) if re_arr.size else 0.0,
        "ERE": float(np.mean(ere_arr)) if ere_arr.size else 0.0,
        "ERE_std": float(np.std(ere_arr)) if ere_arr.size else 0.0,
        "P95": compute_p95(re_arr),
        "SDR": compute_sdr(re_arr),
        "per_landmark_MRE": [float(np.mean(v)) if v else 0.0 for v in per_landmark_mre_values],
        "per_landmark_MRE_std": [float(np.std(v)) if v else 0.0 for v in per_landmark_mre_values],
        "per_landmark_P95": [compute_p95(np.asarray(v)) for v in per_landmark_mre_values],
        "per_landmark_ERE": [float(np.mean(v)) if v else 0.0 for v in per_landmark_ere_values],
        "sample_ids": sample_ids,
        "sample_ere": sample_ere_per_sample,
        "sample_mre": sample_mre_per_sample,
        "pred_landmarks": pred_landmarks_all,
        "radial_errors": re_arr.tolist(),
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _write_split_metrics_txt(
            split_name=split_name,
            metrics=results,
            output_path=os.path.join(save_dir, f"{split_name}_metrics.txt"),
        )
        plot_split_overview(
            results, os.path.join(save_dir, f"{split_name}_metrics_overview.png"),
            title=f"{split_name.upper()} metrics"
        )

        if grid_samples > 0 and vis_names:
            n = min(grid_samples, len(vis_names))
            all_images = np.concatenate(vis_images, axis=0)[:n]
            all_gt = np.concatenate(vis_gt_landmarks, axis=0)[:n]
            all_heatmaps = np.concatenate(vis_heatmaps, axis=0)[:n]
            all_scaling = np.concatenate(vis_scaling, axis=0)[:n]
            all_pred = np.asarray(vis_pred_landmarks[:n])

            plot_prediction_debug_grid(
                images=all_images, gt_landmarks=all_gt, pred_landmarks=all_pred,
                heatmaps=all_heatmaps, sample_ids=vis_names[:n],
                scaling_factors=all_scaling,
                save_path=os.path.join(save_dir, f"{split_name}_prediction_grid.png"),
                title=f"{split_name.upper()} predictions",
            )

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from shared.config import apply_overrides, load_config, set_seed, setup_device
    from shared.data import load_dataset_splits, sorted_records_by_id
    from shared.model import create_landmark_model

    parser = argparse.ArgumentParser(description="Evaluate a trained landmark detection model")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--save-dir", default=None, help="Directory to save evaluation results")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Split to evaluate on")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, mode="eval")
    apply_overrides(cfg, args.override)
    set_seed(cfg["seed"])
    device = setup_device(cfg["gpu"])

    splits, num_landmarks = load_dataset_splits(cfg)
    records = sorted_records_by_id(splits[args.split])

    model = create_landmark_model(cfg=cfg, num_landmarks=num_landmarks, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {args.checkpoint}")

    results = evaluate_model(
        model=model, records=records, cfg=cfg, device=device,
        num_landmarks=num_landmarks, seed=cfg["seed"],
        save_dir=args.save_dir, split_name=args.split,
        grid_samples=cfg["grid_samples"],
    )
    print(f"\n{args.split.upper()} MRE: {results['MRE']:.4f} ± {results['MRE_std']:.4f}")
    print(f"{args.split.upper()} P95: {results['P95']:.4f}")
    print(f"{args.split.upper()} ERE: {results['ERE']:.4f}")
    for thresh, sdr in results["SDR"].items():
        print(f"  SDR@{thresh}: {sdr:.2f}%")
