"""Train the shared base detector (oracle-labeled only) for PACER.

Protocol (spec §5):
  1. Seed 42, GPU from config.
  2. Load splits; pick 10 oracle ids via `select_initial_labeled(seed=42)`.
  3. AdamW(base_learning_rate, base_weight_decay), no scheduler.
  4. Up to base_epochs with early stopping on val_loss (patience 15).
  5. Persist best-val-loss checkpoint + base_metrics.json + ids + run_config
     + training curves.
  6. Idempotent: exit 0 if base_checkpoint.pt already exists, unless --force.
"""

import argparse
import copy
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import apply_overrides, load_config, save_json, set_seed, setup_device
from shared.data import create_dataloader, load_dataset_splits, sorted_records_by_id, subset_records
from shared.metrics import CustomNLLLoss
from shared.model import create_landmark_model
from shared.train_utils import evaluate_one_epoch, train_one_epoch
from shared.visualization import plot_training_curves
from baseline.evaluate import evaluate_model, strip_metrics
from shared.selection import select_initial_labeled, sorted_ids


def run_base(cfg: Dict[str, Any], force: bool = False) -> None:
    set_seed(cfg["seed"])
    device = setup_device(cfg["gpu"])

    splits, num_landmarks = load_dataset_splits(cfg)
    train_records = sorted_records_by_id(splits["train"])
    val_records = sorted_records_by_id(splits["val"])
    test_records = sorted_records_by_id(splits["test"])
    train_by_id = {r["sample_id"]: r for r in train_records}
    train_ids = [r["sample_id"] for r in train_records]

    if cfg["initial_labeled"] >= len(train_ids):
        raise ValueError(
            f"initial_labeled={cfg['initial_labeled']} must be smaller than train size={len(train_ids)}"
        )

    initial_labeled_ids = select_initial_labeled(
        train_ids=train_ids, n_init=cfg["initial_labeled"], seed=cfg["seed"],
    )
    residual_ids = sorted_ids([sid for sid in train_ids if sid not in set(initial_labeled_ids)])

    base_dir = os.path.join(cfg["output_dir"], cfg["dataset"], f"seed_{cfg['seed']}", f"active_{cfg['initial_labeled']}", "base")
    plots_dir = os.path.join(base_dir, "plots")
    checkpoint_path = os.path.join(base_dir, "base_checkpoint.pt")

    if os.path.exists(checkpoint_path) and not force:
        print(f"[train_base] reuse existing {checkpoint_path}")
        return

    os.makedirs(plots_dir, exist_ok=True)

    save_json(os.path.join(base_dir, "run_config.json"), cfg)
    save_json(os.path.join(base_dir, "initial_labeled_ids.json"), initial_labeled_ids)
    save_json(os.path.join(base_dir, "residual_ids.json"), residual_ids)

    model = create_landmark_model(cfg=cfg, num_landmarks=num_landmarks, device=device)

    train_records_subset = subset_records(train_by_id, initial_labeled_ids)
    train_loader = create_dataloader(
        records=train_records_subset, cfg=cfg, shuffle=True, seed=cfg["seed"],
    )
    val_loader = create_dataloader(records=val_records, cfg=cfg, shuffle=False, seed=cfg["seed"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["base_learning_rate"],
        weight_decay=cfg["base_weight_decay"],
    )
    loss_fn = CustomNLLLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_mre": [], "val_ere": []}
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0

    epoch_pbar = tqdm(range(cfg["base_epochs"]), desc="Base train", leave=True)
    for epoch_idx in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_one_epoch(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mre"].append(val_metrics["mre"])
        history["val_ere"].append(val_metrics["ere"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"model_state_dict": best_state, "epoch": epoch_idx + 1, "best_val_loss": best_val_loss},
                checkpoint_path,
            )
            patience = 0
        else:
            patience += 1

        epoch_pbar.set_postfix({
            "train": f"{train_loss:.4f}",
            "val": f"{val_metrics['loss']:.4f}",
            "mre": f"{val_metrics['mre']:.2f}",
            "pat": f"{patience}/{cfg['base_early_stopping_patience']}",
        })

        if patience >= cfg["base_early_stopping_patience"]:
            tqdm.write(f"  Early stopping at epoch {epoch_idx + 1}")
            break

    if best_state is None:
        raise RuntimeError("No epoch completed; best_state is None")
    model.load_state_dict(best_state)

    plot_training_curves(
        history=history,
        save_path=os.path.join(plots_dir, "training_curves.png"),
        title=f"Base training ({cfg['dataset']})",
    )
    save_json(os.path.join(base_dir, "training_history.json"), history)

    val_metrics_full = evaluate_model(
        model=model, records=val_records, cfg=cfg, device=device,
        num_landmarks=num_landmarks, seed=cfg["seed"],
        save_dir=base_dir, split_name="val", grid_samples=cfg["grid_samples"],
    )
    test_metrics_full = evaluate_model(
        model=model, records=test_records, cfg=cfg, device=device,
        num_landmarks=num_landmarks, seed=cfg["seed"],
        save_dir=base_dir, split_name="test", grid_samples=cfg["grid_samples"],
    )

    save_json(
        os.path.join(base_dir, "base_metrics.json"),
        {"val": strip_metrics(val_metrics_full), "test": strip_metrics(test_metrics_full)},
    )
    print(f"[train_base] done. best_val_loss={best_val_loss:.4f}  test_MRE={test_metrics_full['MRE']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PACER base detector")
    parser.add_argument("--config", default="pacer/configs/base.json", help="Path to base config JSON")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    args = parser.parse_args()

    cfg = load_config(args.config, mode="base")
    apply_overrides(cfg, args.override)
    run_base(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
