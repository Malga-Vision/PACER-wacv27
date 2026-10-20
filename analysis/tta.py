"""11-view test-time-augmentation (TTA) disagreement, in physical units.

Label-free uncertainty baseline of the ranking-quality analysis
(``analysis/ranking_quality.py``, Fig. ranking / Tab. corr, "TTA disag.").
Each image is passed through a fixed bank of 11 views (identity, brightness
+/-0.15, gamma 0.7/1.4, rotations +/-5 deg, translations +/-5 px along x and y);
the argmax landmarks of every view are mapped back to the original frame and
the disagreement of a landmark is the mean scaled distance of the 11
predictions from their centroid. The sample score is the mean over landmarks.
"""

import math
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from shared.data import read_image
from shared.metrics import CustomNLLLoss, compute_scaled_distance, extract_landmark

TTA_ROTATION_ANGLES_DEG: Tuple[float, ...] = (5.0, -5.0)
TTA_TRANSLATION_PIXELS: Tuple[Tuple[int, int], ...] = ((5, 0), (-5, 0), (0, 5), (0, -5))


def _apply_affine_tensor(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta.unsqueeze(0), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _rotation_sampling_theta(alpha_rad: float, dtype, device) -> torch.Tensor:
    c, s = math.cos(alpha_rad), math.sin(alpha_rad)
    return torch.tensor([[c, s, 0.0], [-s, c, 0.0]], dtype=dtype, device=device)


def _translation_sampling_theta(tx: int, ty: int, width: int, height: int, dtype, device) -> torch.Tensor:
    tx_norm = 2.0 * float(tx) / max(width, 1)
    ty_norm = 2.0 * float(ty) / max(height, 1)
    return torch.tensor([[1.0, 0.0, -tx_norm], [0.0, 1.0, -ty_norm]], dtype=dtype, device=device)


def tta_views(image: torch.Tensor) -> List[Tuple[str, torch.Tensor, Dict[str, Any]]]:
    """The fixed 11-view bank: (name, view tensor, inversion metadata)."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    clamped = image.clamp(0.0, 1.0)
    views = [
        ("base", image, {"type": "identity"}),
        ("brightness_up", (clamped + 0.15).clamp(0.0, 1.0), {"type": "identity"}),
        ("brightness_down", (clamped - 0.15).clamp(0.0, 1.0), {"type": "identity"}),
        ("gamma_bright", clamped.pow(0.7), {"type": "identity"}),
        ("gamma_dark", clamped.pow(1.4), {"type": "identity"}),
    ]
    for alpha_deg in TTA_ROTATION_ANGLES_DEG:
        alpha_rad = math.radians(float(alpha_deg))
        theta = _rotation_sampling_theta(alpha_rad, image.dtype, image.device)
        views.append((f"rot_{int(alpha_deg):+d}", _apply_affine_tensor(image, theta),
                      {"type": "rotate", "alpha_rad": alpha_rad}))
    for tx, ty in TTA_TRANSLATION_PIXELS:
        theta = _translation_sampling_theta(tx, ty, width, height, image.dtype, image.device)
        views.append((f"trans_{tx:+d}_{ty:+d}", _apply_affine_tensor(image, theta),
                      {"type": "translate", "tx": int(tx), "ty": int(ty)}))
    return views


def _invert_view_landmarks(meta: Dict[str, Any], landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map (x, y) landmarks predicted in a view back to the original image frame."""
    kind = meta.get("type", "identity")
    if kind == "rotate":
        alpha = float(meta["alpha_rad"])
        c, s = math.cos(alpha), math.sin(alpha)
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        dx, dy = landmarks[:, 0] - cx, landmarks[:, 1] - cy
        return np.stack([c * dx + s * dy + cx, -s * dx + c * dy + cy], axis=-1)
    if kind == "translate":
        inverted = landmarks.copy()
        inverted[:, 0] -= float(meta["tx"])
        inverted[:, 1] -= float(meta["ty"])
        return inverted
    return landmarks


def compute_tta_disagreement(model: nn.Module, records: List[Dict[str, Any]], cfg: Dict[str, Any],
                             device: torch.device) -> Dict[str, Dict[str, Any]]:
    """Per-sample TTA disagreement (mm, or px for unscaled datasets).

    Returns {sample_id: {"sample_mean_mm": float, "per_landmark_mm": [float]}}.
    """
    model.eval()
    # same eval-time preprocessing as shared.data.LandmarkDataset (no augmentation)
    transform = A.Compose([
        A.LongestMaxSize(max_size=max(cfg["image_size"])),
        A.PadIfNeeded(min_height=cfg["image_size"][0], min_width=cfg["image_size"][1],
                      border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(normalization="min_max"),
        ToTensorV2(),
    ])
    results: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for record in records:
            image = read_image(record["image_path"], cfg["image_channels"])
            x = transform(image=image)["image"].unsqueeze(0).to(device)
            height, width = int(x.shape[-2]), int(x.shape[-1])
            scaling = np.asarray(record["physical_scaling_factor"], dtype=np.float64)

            preds = []
            for _name, view, meta in tta_views(x):
                probs = CustomNLLLoss.two_d_softmax(model(view)).cpu().numpy()[0]
                lms = np.asarray([extract_landmark(probs[k]) for k in range(probs.shape[0])], dtype=np.float64)
                preds.append(_invert_view_landmarks(meta, lms, width, height))

            stacked = np.stack(preds, axis=0)          # (views, K, 2)
            centroid = stacked.mean(axis=0)
            per_landmark = [float(np.mean([compute_scaled_distance(stacked[t, k], centroid[k], scaling)
                                           for t in range(stacked.shape[0])]))
                            for k in range(stacked.shape[1])]
            results[record["sample_id"]] = {
                "sample_mean_mm": float(np.mean(per_landmark)) if per_landmark else 0.0,
                "per_landmark_mm": per_landmark,
            }
    return results
