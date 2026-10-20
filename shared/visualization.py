"""Plot helpers reused by train_base.py and evaluate.py."""

import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.metrics import compute_scaled_distance


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _to_display_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in {1, 3}:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return np.clip(img, 0.0, 1.0)


def _sorted_sdr_items(sdr: Dict[str, Any]) -> List[Tuple[float, float]]:
    items = [(float(str(k).replace("mm", "").strip()), float(v)) for k, v in sdr.items()]
    items.sort(key=lambda x: x[0])
    return items


def plot_training_curves(
    history: Dict[str, Sequence[float]], save_path: str, title: str = "Training Curves"
) -> None:
    """Train + val loss per epoch. Spec Viz 8."""
    train_loss = np.asarray(history.get("train_loss", []), dtype=np.float64)
    if train_loss.size == 0:
        return
    val_loss = np.asarray(history.get("val_loss", []), dtype=np.float64)
    epochs = np.arange(1, train_loss.size + 1)

    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, "b-", lw=2, label="Train")
    if val_loss.size == train_loss.size:
        ax.plot(epochs, val_loss, "r-", lw=2, label="Val")
    ax.set(xlabel="Epoch", ylabel="Loss", title=title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_overview(metrics: Dict[str, Any], save_path: str, title: str) -> None:
    _ensure_parent(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    global_keys = ["MRE", "ERE", "P95"]
    global_values = [float(metrics.get(k, 0.0)) for k in global_keys]
    bars = axes[0, 0].bar(
        global_keys,
        global_values,
        color=["#2a9d8f", "#e76f51", "#457b9d"],
        alpha=0.85,
    )
    for bar, value in zip(bars, global_values):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() * 0.5,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    loss_value = float(metrics.get("loss", 0.0))
    sample_count = len(metrics.get("sample_ids", []))
    axes[0, 0].text(
        0.98,
        0.97,
        "\n".join(
            [
                f"Loss: {loss_value:.4f}",
                f"MRE std: {float(metrics.get('MRE_std', 0.0)):.3f}",
                f"ERE std: {float(metrics.get('ERE_std', 0.0)):.3f}",
                f"N: {sample_count}",
            ]
        ),
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    axes[0, 0].set(title="Global Metrics", ylabel="Value")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    sdr_items = _sorted_sdr_items(metrics.get("SDR", {}))
    if sdr_items:
        axes[0, 1].plot([x[0] for x in sdr_items], [x[1] for x in sdr_items], "o-", lw=2, ms=6)
        for threshold, sdr_value in sdr_items:
            axes[0, 1].annotate(
                f"{sdr_value:.1f}%",
                xy=(threshold, sdr_value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes[0, 1].set(title="SDR Curve", xlabel="Threshold (mm)", ylabel="SDR (%)")
    axes[0, 1].grid(True, alpha=0.3)

    per_lm_mre = np.asarray(metrics.get("per_landmark_MRE", []), dtype=np.float64)
    if per_lm_mre.size > 0:
        axes[1, 0].bar(np.arange(1, per_lm_mre.size + 1), per_lm_mre, color="#1d3557", alpha=0.85)
        axes[1, 0].axhline(float(np.mean(per_lm_mre)), color="#e76f51", ls="--", lw=1.5,
                           label=f"mean={float(np.mean(per_lm_mre)):.3f}")
        axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].set(title="Per-landmark MRE", xlabel="Landmark", ylabel="MRE (mm)")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    per_lm_ere = np.asarray(metrics.get("per_landmark_ERE", []), dtype=np.float64)
    if per_lm_ere.size > 0:
        axes[1, 1].bar(np.arange(1, per_lm_ere.size + 1), per_lm_ere, color="#f4a261", alpha=0.85)
        axes[1, 1].axhline(float(np.mean(per_lm_ere)), color="#264653", ls="--", lw=1.5,
                           label=f"mean={float(np.mean(per_lm_ere)):.3f}")
        axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].set(title="Per-landmark ERE", xlabel="Landmark", ylabel="ERE (mm)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_debug_grid(
    images: np.ndarray,
    gt_landmarks: np.ndarray,
    pred_landmarks: np.ndarray,
    heatmaps: np.ndarray,
    sample_ids: Sequence[str],
    scaling_factors: np.ndarray,
    save_path: str,
    title: str,
    error_threshold_mm: float = 2.0,
) -> None:
    if len(sample_ids) == 0:
        return

    num_samples = len(sample_ids)
    cols = min(4, num_samples)
    rows = 2 * int(np.ceil(num_samples / cols))

    _ensure_parent(save_path)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_np = np.asarray(axes)
    if axes_np.ndim == 1:
        axes_np = axes_np.reshape(rows, 1)

    for r in range(rows):
        for c in range(cols):
            axes_np[r, c].axis("off")

    for idx in range(num_samples):
        col = idx % cols
        row_block = (idx // cols) * 2

        image = _to_display_image(images[idx])
        gt = np.asarray(gt_landmarks[idx], dtype=np.float64)
        pred = np.asarray(pred_landmarks[idx], dtype=np.float64)
        scaling = np.asarray(scaling_factors[idx], dtype=np.float64)
        heatmap_all = np.asarray(heatmaps[idx], dtype=np.float32)

        hmap_ax = axes_np[row_block, col]
        lm_ax = axes_np[row_block + 1, col]

        hmap_ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        peak = np.max(heatmap_all, axis=(1, 2), keepdims=True)
        peak[peak == 0.0] = 1.0
        fused = np.max(heatmap_all / peak, axis=0)
        hmap_ax.imshow(fused, cmap="inferno", alpha=0.55)
        hmap_ax.set_title(f"{sample_ids[idx]} | heatmaps", fontsize=9)
        hmap_ax.axis("off")

        lm_ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        lm_ax.scatter(gt[:, 0], gt[:, 1], c="blue", s=18, label="GT", zorder=2)
        for lm in range(gt.shape[0]):
            err_mm = compute_scaled_distance(pred[lm], gt[lm], scaling)
            color = "green" if err_mm <= error_threshold_mm else "red"
            lm_ax.scatter(pred[lm, 0], pred[lm, 1], c=color, s=20, marker="x", zorder=3)
            lm_ax.plot([gt[lm, 0], pred[lm, 0]], [gt[lm, 1], pred[lm, 1]],
                       color="yellow", lw=0.8, alpha=0.6, zorder=1)
        if idx == 0:
            lm_ax.scatter([], [], c="green", s=20, marker="x", label=f"Pred <= {error_threshold_mm:.1f}mm")
            lm_ax.scatter([], [], c="red", s=20, marker="x", label=f"Pred > {error_threshold_mm:.1f}mm")
            lm_ax.legend(loc="upper right", fontsize=7)
        lm_ax.set_title(f"{sample_ids[idx]} | GT vs pred", fontsize=9)
        lm_ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
