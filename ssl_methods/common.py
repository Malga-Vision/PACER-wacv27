"""Shared utilities for the semi-supervised (SSL) baselines.

All baselines share the same controlled setup as PACER (the paper method):
  * U-Net / ResNet-101 detector, 2D-softmax heatmaps, NLL objective;
  * finetune from the shared base checkpoint of the (dataset, seed, budget) triple;
  * oracle = the labeled set used by the base detector; residual = unlabeled pool;
  * matched budget E epochs, lr = finetune_learning_rate, batch_size from config.

Consistency design (shared by FixMatch / FlexMatch / FreeMatch):
  weak view  = canonical resize+pad+normalize (no geometric aug);
  teacher (or the model itself) predicts on the weak view -> decode per-landmark
  (x,y) + peak-softmax confidence; decoded coordinates become pseudo-labels;
  strong view = the project's ``advanced`` augmentation applied *jointly* to the
  canonical image and pseudo-coordinates, so the target heatmap is geometrically
  consistent with the strongly-augmented student input.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import apply_overrides, load_config, save_json, set_seed, setup_device
from shared.data import (
    LandmarkDataset,
    create_dataloader,
    load_dataset_splits,
    read_image,
    sorted_records_by_id,
    subset_records,
)
from baseline.evaluate import evaluate_model, strip_metrics
from shared.metrics import CustomNLLLoss, extract_landmark, keypoints2heatmaps
from shared.model import create_landmark_model
from shared.selection import select_initial_labeled, sorted_ids

CONFIG_FINETUNE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pacer", "configs", "finetune.json")
RESULTS_ROOT = "/path/to/results"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=["chest", "cephalo", "hand"])
    parser.add_argument("--initial_labeled", required=True, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=120, type=int, help="matched budget E")
    parser.add_argument("--output_dir", default=RESULTS_ROOT,
                        help="results root holding the base runs; set it (the default is a placeholder)")
    parser.add_argument("--lambda_u", default=1.0, type=float, help="max unlabeled loss weight")
    parser.add_argument("--rampup_frac", default=0.1, type=float, help="fraction of epochs for consistency ramp-up")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")


def make_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    """Load the finetune config and override the run-specific keys."""
    cfg = load_config(CONFIG_FINETUNE, mode="finetune")
    cfg["dataset"] = args.dataset
    cfg["initial_labeled"] = int(args.initial_labeled)
    cfg["gpu"] = int(args.gpu)
    cfg["seed"] = int(args.seed)
    cfg["output_dir"] = args.output_dir
    apply_overrides(cfg, args.override)
    return cfg


def ssl_run_dir(cfg: Dict[str, Any], method: str) -> str:
    return os.path.join(
        cfg["output_dir"], cfg["dataset"], f"seed_{cfg['seed']}",
        f"active_{cfg['initial_labeled']}", "ssl", method,
    )


def resolve_oracle_residual(cfg: Dict[str, Any], train_ids: List[str], base_dir: str) -> Tuple[List[str], List[str], str]:
    il_path = os.path.join(base_dir, "initial_labeled_ids.json")
    res_path = os.path.join(base_dir, "residual_ids.json")
    if os.path.exists(il_path) and os.path.exists(res_path):
        with open(il_path) as h:
            oracle_ids = sorted_ids(list(json.load(h)))
        with open(res_path) as h:
            residual_ids = sorted_ids(list(json.load(h)))
        source = "base_artifacts"
    else:
        oracle_ids = select_initial_labeled(train_ids=train_ids, n_init=cfg["initial_labeled"], seed=cfg["seed"])
        residual_ids = sorted_ids([s for s in train_ids if s not in set(oracle_ids)])
        source = "deterministic_from_seed"
    overlap = set(oracle_ids) & set(residual_ids)
    if overlap:
        raise ValueError(f"oracle/residual overlap: {len(overlap)}")
    return oracle_ids, residual_ids, source


class AugViews:
    """Thin wrapper over LandmarkDataset to reuse its transforms.

    weak       = canonical resize+pad+normalize (no aug);
    strong     = ``advanced`` level (geometric+intensity) applied jointly to image+coords;
    intensity  = intensity/noise aug at fixed geometry (used by Mean Teacher).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._ds = LandmarkDataset(
            records=[], dataset_name=cfg["dataset"], image_size=cfg["image_size"],
            sigma=cfg["sigma"], training_mode=True, channels=cfg["image_channels"],
            pseudo_labels=None,
        )
        self.image_size = tuple(int(v) for v in cfg["image_size"])
        self.sigma = float(cfg["sigma"])
        self._normalize = A.Compose(self._ds._normalize_transforms)
        self._intensity = A.Compose(self._ds._intensity_augmentations() + self._ds._normalize_transforms)

    def canonical_uint8(self, image_np: np.ndarray) -> np.ndarray:
        return self._ds._resize_pad_image_only(image=image_np)["image"]

    def weak_tensor(self, canon_uint8: np.ndarray) -> torch.Tensor:
        return self._normalize(image=canon_uint8)["image"].float()

    def intensity_tensor(self, canon_uint8: np.ndarray) -> torch.Tensor:
        return self._intensity(image=canon_uint8)["image"].float()

    def strong(self, canon_uint8: np.ndarray, coords: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        img_t, lms = self._ds._apply_with_fallback(canon_uint8, coords, skip_resize_pad=True)
        return img_t.float(), lms

    def heatmap(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=np.float64).copy()
        c[:, 0] = np.clip(c[:, 0], 0, self.image_size[1] - 1)
        c[:, 1] = np.clip(c[:, 1], 0, self.image_size[0] - 1)
        return keypoints2heatmaps(landmarks=c, img_size=self.image_size, sigma=self.sigma)


class UnlabeledDataset(Dataset):
    """Reads image -> canonical uint8; views built in the training loop."""

    def __init__(self, records: List[Dict[str, Any]], cfg: Dict[str, Any], views: AugViews) -> None:
        self.records = records
        self.channels = int(cfg["image_channels"])
        self.views = views

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        img = read_image(r["image_path"], channels=self.channels)
        canon = self.views.canonical_uint8(img)
        return {"name": r["sample_id"], "canon": torch.from_numpy(np.ascontiguousarray(canon))}


def make_unlabeled_loader(records, cfg, views, seed):
    from shared.config import make_torch_generator, seed_worker
    ds = UnlabeledDataset(records, cfg, views)
    return DataLoader(
        ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]), drop_last=False, worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
    )


def decode_probs(probs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(B, K, H, W) softmax probabilities -> coords (B, K, 2), peak-confidence (B, K)."""
    B, K = probs_np.shape[0], probs_np.shape[1]
    coords = np.zeros((B, K, 2), dtype=np.float64)
    conf = np.zeros((B, K), dtype=np.float64)
    for b in range(B):
        for k in range(K):
            hm = probs_np[b, k]
            coords[b, k] = extract_landmark(hm)
            conf[b, k] = float(hm.max())
    return coords, conf


def two_d_softmax(logits: torch.Tensor) -> torch.Tensor:
    return CustomNLLLoss.two_d_softmax(logits)


@torch.no_grad()
def calibrate_tau(model, oracle_records, cfg, views, device, quantile=0.5):
    """Base confidence threshold for FixMatch / FlexMatch, grounded in the labeled set."""
    model.train()
    bs = max(2, int(cfg["batch_size"]))
    tensors = [views.weak_tensor(views.canonical_uint8(read_image(r["image_path"], channels=int(cfg["image_channels"]))))
               for r in oracle_records]
    confs = []
    for i in range(0, len(tensors), bs):
        chunk = tensors[i:i + bs]
        if len(chunk) == 1 and len(tensors) > 1:
            chunk = tensors[max(0, i - 1):i + 1]
        w = torch.stack(chunk).to(device)
        p = two_d_softmax(model(w)).cpu().numpy()
        _, c = decode_probs(p)
        confs.append(c.reshape(-1))
    allc = np.concatenate(confs) if confs else np.asarray([0.0])
    return float(np.quantile(allc, quantile))


def masked_unsup_loss(per_kp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1.0)
    return ((per_kp * mask).sum(dim=1) / denom).mean()


@torch.no_grad()
def ema_update(teacher, student, decay):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.data.copy_(sb.data)


def clone_model(model):
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def sigmoid_rampup(current, length):
    if length <= 0:
        return 1.0
    phase = 1.0 - max(0.0, min(1.0, current / length))
    return float(math.exp(-5.0 * phase * phase))


def supervised_loss(model, batch, loss_fn, device):
    images = batch["image"].to(device)
    targets = batch["target"].to(device)
    sample_weight = batch["sample_weight"].to(device)
    landmark_mask = batch["landmark_mask"].to(device)
    probs = two_d_softmax(model(images))
    per_kp = loss_fn(probs, targets)
    masked_sum = (per_kp * landmark_mask).sum(dim=1)
    denom = landmark_mask.sum(dim=1).clamp_min(1)
    return ((masked_sum / denom) * sample_weight).mean()


def setup(cfg):
    set_seed(cfg["seed"])
    device = setup_device(cfg["gpu"])
    splits, num_landmarks = load_dataset_splits(cfg)
    train_records = sorted_records_by_id(splits["train"])
    val_records = sorted_records_by_id(splits["val"])
    test_records = sorted_records_by_id(splits["test"])
    train_by_id = {r["sample_id"]: r for r in train_records}
    train_ids = [r["sample_id"] for r in train_records]

    base_root = cfg.get("base_root", cfg["output_dir"])
    base_dir = os.path.join(base_root, cfg["dataset"], f"seed_{cfg['seed']}", f"active_{cfg['initial_labeled']}", "base")
    ckpt_path = os.path.join(base_dir, "base_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing base checkpoint {ckpt_path}")

    oracle_ids, residual_ids, src = resolve_oracle_residual(cfg, train_ids, base_dir)
    oracle_records = subset_records(train_by_id, oracle_ids)
    residual_records = subset_records(train_by_id, residual_ids)

    model = create_landmark_model(cfg=cfg, num_landmarks=num_landmarks, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    views = AugViews(cfg)
    labeled_loader = create_dataloader(records=oracle_records, cfg=cfg, shuffle=True, seed=cfg["seed"])
    unlabeled_loader = make_unlabeled_loader(residual_records, cfg, views, seed=cfg["seed"])
    val_loader = create_dataloader(records=val_records, cfg=cfg, shuffle=False, seed=cfg["seed"])

    return {
        "device": device, "num_landmarks": int(num_landmarks),
        "val_records": val_records, "test_records": test_records,
        "oracle_records": oracle_records, "residual_records": residual_records,
        "model": model, "views": views, "base_dir": base_dir, "ckpt_path": ckpt_path,
        "labeled_loader": labeled_loader, "unlabeled_loader": unlabeled_loader,
        "val_loader": val_loader, "oracle_ids_source": src,
        "oracle_ids": oracle_ids, "residual_ids": residual_ids,
    }


def labeled_cycle(loader):
    while True:
        for b in loader:
            yield b


def finalize(method, cfg, ctx, history, extra_meta):
    run_dir = ssl_run_dir(cfg, method)
    os.makedirs(os.path.join(run_dir, "eval_final"), exist_ok=True)
    model, device = ctx["model"], ctx["device"]

    torch.save({"model_state_dict": model.state_dict()}, os.path.join(run_dir, "final_checkpoint.pt"))
    save_json(os.path.join(run_dir, "training_history.json"), history)
    save_json(os.path.join(run_dir, "run_config.json"), {
        "method": method, "run_dir": run_dir, "epochs": int(cfg.get("epochs", len(history))),
        "checkpoint_path": ctx["ckpt_path"], "oracle_ids_source": ctx["oracle_ids_source"],
        "n_oracle": len(ctx["oracle_ids"]), "n_residual": len(ctx["residual_ids"]),
        **extra_meta, **cfg,
    })

    val_res = evaluate_model(model=model, records=ctx["val_records"], cfg=cfg, device=device,
                             num_landmarks=ctx["num_landmarks"], seed=cfg["seed"],
                             save_dir=os.path.join(run_dir, "eval_final"), split_name="val",
                             grid_samples=cfg["grid_samples"])
    test_res = evaluate_model(model=model, records=ctx["test_records"], cfg=cfg, device=device,
                              num_landmarks=ctx["num_landmarks"], seed=cfg["seed"],
                              save_dir=os.path.join(run_dir, "eval_final"), split_name="test",
                              grid_samples=cfg["grid_samples"])
    final_metrics = {"val": strip_metrics(val_res), "test": strip_metrics(test_res)}
    save_json(os.path.join(run_dir, "metrics_final.json"), final_metrics)
    print(f"[{method}] done. run_dir={run_dir} test_MRE={final_metrics['test']['MRE']:.3f} "
          f"test_ERE={final_metrics['test']['ERE']:.3f}", flush=True)
    return final_metrics