"""Sample scoring and percentile-gated pseudo-label selection.

The residual pool is scored by Expected Radial Error (ERE), a heatmap-derived
uncertainty estimate. Each curriculum stage admits the bottom-percentile of
residual samples ranked by mean ERE ("easy first") and packages their
predictions as pseudo-labels.
"""

import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data import create_dataloader
from shared.metrics import CustomNLLLoss, calculate_ere, extract_landmark


def score_pool(
    model,
    records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device,
) -> Dict[str, Dict[str, Any]]:
    """Score every sample with per-landmark ERE and predicted landmarks.

    Output rows: ``{"sample_ere", "per_landmark_ere", "pred_landmarks"}`` where
    ``sample_ere`` is the mean of ``per_landmark_ere``.
    """
    import torch

    dataloader = create_dataloader(records=records, cfg=cfg, shuffle=False, seed=int(cfg["seed"]))
    score_table: Dict[str, Dict[str, Any]] = {}

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            scaling = batch["physical_scaling_factor"].cpu().numpy()
            names = batch["name"]

            probs = CustomNLLLoss.two_d_softmax(model(images)).cpu().numpy()

            for i in range(probs.shape[0]):
                sample_id = str(names[i])
                sample_heatmaps = probs[i]
                per_landmark_ere: List[float] = []
                pred_landmarks: List[List[float]] = []

                for lm in range(sample_heatmaps.shape[0]):
                    heatmap = sample_heatmaps[lm]
                    pred = extract_landmark(heatmap)
                    ere = calculate_ere(heatmap, pred * scaling[i], scaling[i])
                    pred_landmarks.append(pred.tolist())
                    per_landmark_ere.append(float(ere))

                score_table[sample_id] = {
                    "sample_ere": float(np.mean(per_landmark_ere)),
                    "pred_landmarks": pred_landmarks,
                    "per_landmark_ere": per_landmark_ere,
                }

    return score_table


def select_by_percentile(
    score_table: Dict[str, Dict[str, Any]],
    percentile: float,
) -> List[str]:
    """Admit the ``percentile``% of samples with the lowest mean ERE.

    ``percentile`` is in [0, 100]: 0 admits no sample, 100 admits all. The gate
    is adaptive -- it selects the bottom-k fraction of the *current* ERE
    distribution rather than a fixed millimetre threshold, so the effective
    threshold tightens automatically as the model improves.
    """
    p = float(percentile)
    if p < 0.0 or p > 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")

    scored = []
    for sid, s in score_table.items():
        per_lm = s.get("per_landmark_ere") or []
        if not per_lm:
            continue
        mean_ere = float(np.mean(per_lm))
        scored.append((mean_ere, sid))

    if not scored:
        return []

    scored.sort()  # ascending mean ERE -> easiest samples first
    k = int(round(len(scored) * p / 100.0))
    k = max(0, min(k, len(scored)))
    return [sid for _, sid in scored[:k]]


def select_random_by_percentile(
    score_table: Dict[str, Dict[str, Any]],
    percentile: float,
    seed: int,
) -> List[str]:
    """Random-admission ablation (selection-signal control).

    Admits the *same number* ``k = round(N * percentile/100)`` of residual
    samples as :func:`select_by_percentile`, but drawn uniformly at random
    (ignoring ERE). This isolates the contribution of the ERE *ranking* at a
    matched curriculum and per-stage budget. The draw is deterministic in
    ``seed`` (finetune derives it from the run seed and the stage index).
    """
    p = float(percentile)
    if p < 0.0 or p > 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")

    valid = sorted(sid for sid, s in score_table.items() if s.get("per_landmark_ere"))
    if not valid:
        return []
    k = int(round(len(valid) * p / 100.0))
    k = max(0, min(k, len(valid)))
    if k == 0:
        return []
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.asarray(valid), size=k, replace=False)
    return sorted(chosen.tolist())


def build_per_landmark_percentile_masks(
    score_table: Dict[str, Dict[str, Any]],
    percentile: float,
) -> Dict[str, List[int]]:
    """Per-landmark admission masks under the same percentile gate.

    The image-level gate admits the bottom-``percentile``% of *samples* by mean
    ERE. The per-landmark analogue admits the bottom-``percentile``% of all
    individual landmark predictions (across every residual sample) by their own
    ERE: it pools every ``(sample, landmark)`` ERE, takes the lowest
    ``k = round(total_landmarks * percentile/100)``, and sets ``mask[sid][k]=1``
    for each admitted landmark. A sample with no admitted landmark gets an
    all-zero mask (and is dropped from the train set upstream).
    """
    p = float(percentile)
    if p < 0.0 or p > 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")

    flat = []  # (ere, sid, landmark_idx)
    for sid in sorted(score_table):
        per_lm = score_table[sid].get("per_landmark_ere") or []
        for k, ere in enumerate(per_lm):
            flat.append((float(ere), sid, k))

    masks: Dict[str, List[int]] = {
        sid: [0] * len(score_table[sid].get("per_landmark_ere") or [])
        for sid in score_table
    }
    if not flat:
        return masks

    flat.sort()  # ascending ERE -> easiest landmarks first
    n_admit = int(round(len(flat) * p / 100.0))
    n_admit = max(0, min(n_admit, len(flat)))
    for _, sid, k in flat[:n_admit]:
        masks[sid][k] = 1
    return masks


def build_pseudo_labels(
    score_table: Dict[str, Dict[str, Any]],
    selected_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Package admitted samples as pseudo-labels (predicted landmarks only).

    Non-admitted samples are excluded upstream and are not packaged. Used by the
    image-level gate (and the random-admission ablation), where every landmark
    of an admitted sample is trained on (implicit all-ones mask in data.py).
    """
    return {
        sid: {"landmarks": score_table[sid]["pred_landmarks"]}
        for sid in selected_ids
    }


def build_pseudo_labels_per_landmark(
    score_table: Dict[str, Dict[str, Any]],
    landmark_masks: Dict[str, List[int]],
) -> Dict[str, Dict[str, Any]]:
    """Package per-landmark-gated pseudo-labels.

    Includes only samples with at least one admitted landmark; each carries its
    predicted landmarks plus the binary ``landmark_mask`` so excluded landmarks
    contribute no loss (see the masked path in ``train_utils.train_one_epoch``
    and ``data.py``'s ``_pseudo_item``).
    """
    pseudo: Dict[str, Dict[str, Any]] = {}
    for sid, mask in landmark_masks.items():
        if any(mask):
            pseudo[sid] = {
                "landmarks": score_table[sid]["pred_landmarks"],
                "landmark_mask": list(map(int, mask)),
            }
    return pseudo
