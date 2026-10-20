"""Shared epoch-level training and validation loops."""

import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.metrics import CustomNLLLoss, summarize_sample_errors


def train_one_epoch(model, loader, optimizer, device) -> float:
    """Masked+weighted per-keypoint NLL over one epoch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    nll = CustomNLLLoss(reduction="perkeypoint")
    running = 0.0
    for batch in tqdm(loader, desc="Train batches", leave=False):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        landmark_mask = batch["landmark_mask"].to(device)

        probs = CustomNLLLoss.two_d_softmax(model(images))
        per_kp = nll(probs, targets)
        masked = (per_kp * landmark_mask).sum(dim=1)
        denom = landmark_mask.sum(dim=1).clamp_min(1)
        loss = ((masked / denom) * sample_weight).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += float(loss.item())
    return running / max(len(loader), 1)


def evaluate_one_epoch(model, loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    per_landmark_mre: List[float] = []
    per_landmark_ere: List[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val batches", leave=False):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            gt_landmarks = batch["landmarks"].cpu().numpy()
            scaling = batch["physical_scaling_factor"].cpu().numpy()

            probs = CustomNLLLoss.two_d_softmax(model(images))
            losses.append(float(loss_fn(probs, targets).item()))
            probs_np = probs.cpu().numpy()
            for i in range(probs_np.shape[0]):
                s = summarize_sample_errors(
                    heatmaps=probs_np[i], gt_landmarks=gt_landmarks[i], scale=scaling[i]
                )
                per_landmark_mre.extend(s["per_landmark_mre"])
                per_landmark_ere.extend(s["per_landmark_ere"])

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "mre": float(np.mean(per_landmark_mre)) if per_landmark_mre else 0.0,
        "ere": float(np.mean(per_landmark_ere)) if per_landmark_ere else 0.0,
    }
