"""All metrics, loss functions, and heatmap utilities for landmark detection."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter


# ─── Constants ────────────────────────────────────────────────────────────────

SDR_THRESHOLDS = (2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 10.0)


# ─── Loss ─────────────────────────────────────────────────────────────────────

class CustomNLLLoss(nn.Module):
    """Negative log-likelihood loss for dense landmark heatmaps.

    Uses a spatial-softmax formulation: ``exp(x) / sum(exp(x))`` over the
    spatial (H, W) dimensions. Supports ``mean`` (batch mean of per-landmark
    sum), ``perkeypoint`` ((B, K) per-landmark sum), and ``none`` (full tensor).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        allowed = {"mean", "none", "perkeypoint"}
        if reduction not in allowed:
            raise ValueError(f"Unsupported reduction '{reduction}', expected one of {sorted(allowed)}")
        self.reduction = reduction

    @staticmethod
    def two_d_softmax(x: torch.Tensor) -> torch.Tensor:
        """Apply spatial softmax independently for each (sample, landmark) pair.

        Subtracts the per-(sample, landmark) spatial max before exp for
        numerical stability. Mathematically equivalent to exp(x)/sum(exp(x)).
        """
        max_per_channel = torch.amax(x, dim=(2, 3), keepdim=True)
        exp_y = torch.exp(x - max_per_channel)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        nll = -target * torch.log(prediction.float() + 1e-10)
        if self.reduction == "none":
            return nll
        if self.reduction == "perkeypoint":
            return torch.sum(nll, dim=(2, 3))
        return torch.mean(torch.sum(nll, dim=(2, 3)))


# ─── Heatmap generation ───────────────────────────────────────────────────────

def keypoints2heatmaps(landmarks: np.ndarray, img_size: Tuple[int, int], sigma: float = 0.0) -> np.ndarray:
    """Generate one binary heatmap per landmark, optionally Gaussian-smoothed."""
    height, width = int(img_size[0]), int(img_size[1])
    heatmaps = np.zeros((landmarks.shape[0], height, width), dtype=np.float32)

    for idx, point in enumerate(landmarks):
        x_coord = int(np.clip(round(float(point[0])), 0, width - 1))
        y_coord = int(np.clip(round(float(point[1])), 0, height - 1))
        heatmaps[idx, y_coord, x_coord] = 1.0

        if sigma > 0:
            heatmaps[idx] = gaussian_filter(heatmaps[idx], sigma=sigma)

        peak = float(heatmaps[idx].max())
        if peak > 0:
            heatmaps[idx] = np.where(heatmaps[idx] < 0.5 * peak, 0.0, 1.0)

    return heatmaps.astype(np.float32)


# ─── Landmark extraction ──────────────────────────────────────────────────────

def _quadratic_subpixel_offset(heatmap: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    max_val = float(heatmap[y, x])
    if x <= 0 or y <= 0 or x >= heatmap.shape[1] - 1 or y >= heatmap.shape[0] - 1:
        return 0.0, 0.0
    denom_x = 2.0 * max_val - float(heatmap[y, x + 1]) - float(heatmap[y, x - 1])
    denom_y = 2.0 * max_val - float(heatmap[y + 1, x]) - float(heatmap[y - 1, x])
    dx = 0.5 * (float(heatmap[y, x + 1]) - float(heatmap[y, x - 1])) / denom_x if denom_x != 0 else 0.0
    dy = 0.5 * (float(heatmap[y + 1, x]) - float(heatmap[y - 1, x])) / denom_y if denom_y != 0 else 0.0
    return dx, dy


def extract_landmark(heatmap: np.ndarray) -> np.ndarray:
    """Convert a landmark heatmap into an (x, y) coordinate via argmax + subpixel refinement."""
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    dx, dy = _quadratic_subpixel_offset(heatmap, int(x), int(y))
    return np.asarray([float(x + dx), float(y + dy)], dtype=np.float64)


def get_mode_probability(heatmap: np.ndarray) -> float:
    return float(np.max(heatmap))


def compute_euclidean_distance(predicted: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(pred - gt))


def compute_scaled_distance(
    predicted_point: np.ndarray,
    target_point: np.ndarray,
    scale: np.ndarray,
) -> float:
    scale_arr = np.asarray(scale, dtype=np.float64)
    return compute_euclidean_distance(
        np.asarray(predicted_point, dtype=np.float64) * scale_arr,
        np.asarray(target_point, dtype=np.float64) * scale_arr,
    )


def summarize_sample_errors(
    heatmaps: np.ndarray,
    gt_landmarks: np.ndarray,
    scale: np.ndarray,
) -> Dict[str, Any]:
    pred_landmarks: List[List[float]] = []
    per_landmark_mre: List[float] = []
    per_landmark_ere: List[float] = []
    mode_probs: List[float] = []

    scale_arr = np.asarray(scale, dtype=np.float64)
    for lm in range(heatmaps.shape[0]):
        heatmap = np.asarray(heatmaps[lm], dtype=np.float32)
        pred = extract_landmark(heatmap)
        gt = np.asarray(gt_landmarks[lm], dtype=np.float64)
        pred_scaled = pred * scale_arr

        per_landmark_mre.append(compute_scaled_distance(pred, gt, scale_arr))
        per_landmark_ere.append(float(calculate_ere(heatmap, pred_scaled, scale_arr)))
        mode_probs.append(get_mode_probability(heatmap))
        pred_landmarks.append(pred.tolist())

    return {
        "pred_landmarks": pred_landmarks,
        "per_landmark_mre": per_landmark_mre,
        "per_landmark_ere": per_landmark_ere,
        "mode_probs": mode_probs,
    }


# ─── Scale factor ─────────────────────────────────────────────────────────────

def get_scale_factor(
    original_image_shape: Tuple[int, int],
    transformed_image_shape: Tuple[int, int],
) -> Tuple[float, float]:
    """Compute inverse resize ratio for longest-side resize with zero-padding."""
    orig_h, orig_w = float(original_image_shape[0]), float(original_image_shape[1])
    out_h, out_w = float(transformed_image_shape[0]), float(transformed_image_shape[1])
    resize_ratio = min(out_h / max(orig_h, 1e-8), out_w / max(orig_w, 1e-8))
    if resize_ratio <= 0:
        raise ValueError("Invalid resize ratio computed from image shapes")
    inv = 1.0 / resize_ratio
    return inv, inv


# ─── Error metrics ────────────────────────────────────────────────────────────

def calculate_ere(
    heatmap: np.ndarray,
    predicted_point_scaled: np.ndarray,
    pixel_size: np.ndarray,
    significant_pixel_cutoff: float = 0.05,
) -> float:
    """Expected Radial Error: probability-weighted mean distance from the predicted point."""
    peak = float(np.max(heatmap))
    if peak <= 0.0:
        return 0.0
    weights = heatmap / peak
    weights = np.where(weights > significant_pixel_cutoff, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return 0.0
    weights = weights / total
    ys, xs = np.mgrid[0: weights.shape[0], 0: weights.shape[1]]
    coords = np.stack([xs, ys], axis=-1).reshape(-1, 2) * pixel_size
    return float(np.dot(np.linalg.norm(coords - predicted_point_scaled, axis=1), weights.reshape(-1)))


def compute_sdr(distance_list: np.ndarray, thresholds: Sequence[float] = SDR_THRESHOLDS) -> Dict[str, float]:
    """Successful Detection Rate in percent for each distance threshold."""
    if np.asarray(distance_list).size == 0:
        return {f"{float(t)}mm": 0.0 for t in thresholds}
    return {
        f"{float(t)}mm": float(100.0 * np.mean(np.asarray(distance_list) <= float(t)))
        for t in thresholds
    }


def compute_p95(radial_errors: np.ndarray) -> float:
    arr = np.asarray(radial_errors, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 95))
