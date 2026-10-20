"""Percentile-gated curriculum finetuning from a base checkpoint.

The residual (unlabeled) pool is scored by Expected Radial Error (ERE) at the
start of every curriculum stage. Each stage admits the bottom-percentile of
residual samples (easiest first) as pseudo-labels and trains on them together
with the constant oracle pool. The percentile schedule is derived from
``num_stages``, ``percentile_cap`` and the optional ramp shape ``schedule_gamma``
(see schedules.py).

If --checkpoint is omitted, finetune uses ``finetune_checkpoint_path`` from the
config when present, else the shared base checkpoint produced by train_base.py.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
from pacer.schedules import build_percentile_schedule, schedule_tag
from shared.selection import select_initial_labeled, sorted_ids
from pacer.selection import (
    build_per_landmark_percentile_masks,
    build_pseudo_labels,
    build_pseudo_labels_per_landmark,
    score_pool,
    select_by_percentile,
    select_random_by_percentile,
)


def _run_dir(cfg: Dict[str, Any], tag: str, variant: str = "percentile") -> str:
    seed_root = os.path.join(cfg["output_dir"], cfg["dataset"], f"seed_{cfg['seed']}")
    active_dir = f"active_{cfg['initial_labeled']}"
    return os.path.join(seed_root, active_dir, tag, variant)


# ─── Curriculum stage bookkeeping ─────────────────────────────────────────────

def _stage_entry(
    stage_idx: int,
    percentile: float,
    start_epoch: int,
    end_epoch: int,
    score_table: Dict[str, Dict[str, Any]],
    selected_ids: List[str],
    previous: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Diagnostics for one curriculum stage.

    Reports the parameterised percentile, the effective millimetre threshold it
    induced (max mean ERE among admitted samples), the admitted fraction, the
    mean ERE of admitted samples, and the Jaccard overlap with the previous
    stage's selection.
    """
    n_total = len(score_table)
    n_selected = len(selected_ids)
    mean_ere: Optional[float] = None
    effective_threshold_mm: Optional[float] = None
    if selected_ids:
        per_sample_means = [
            float(np.mean(score_table[sid]["per_landmark_ere"]))
            for sid in selected_ids
            if score_table[sid].get("per_landmark_ere")
        ]
        if per_sample_means:
            mean_ere = float(np.mean(per_sample_means))
            effective_threshold_mm = float(max(per_sample_means))

    jaccard: Optional[float] = None
    if previous:
        prev_ids = set(previous[-1].get("selected_ids", []))
        curr_ids = set(selected_ids)
        union = prev_ids | curr_ids
        jaccard = float(len(prev_ids & curr_ids) / len(union)) if union else 1.0

    return {
        "stage_idx": int(stage_idx),
        "percentile": float(percentile),
        "effective_threshold_mm": effective_threshold_mm,
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "n_total_residual": int(n_total),
        "n_selected_samples": int(n_selected),
        "fraction_selected": float(n_selected / max(n_total, 1)),
        "mean_selected_sample_ere": mean_ere,
        "jaccard_with_previous": jaccard,
        "selected_ids": list(selected_ids),
    }


def _resolve_oracle_and_residual_ids(
    cfg: Dict[str, Any],
    train_ids: List[str],
    base_dir: str,
) -> Tuple[List[str], List[str], str]:
    import json

    initial_labeled_path = os.path.join(base_dir, "initial_labeled_ids.json")
    residual_path = os.path.join(base_dir, "residual_ids.json")
    if os.path.exists(initial_labeled_path) and os.path.exists(residual_path):
        with open(initial_labeled_path) as handle:
            oracle_ids = sorted_ids(list(json.load(handle)))
        with open(residual_path) as handle:
            residual_ids = sorted_ids(list(json.load(handle)))
        source = "base_artifacts"
    else:
        if cfg["initial_labeled"] >= len(train_ids):
            raise ValueError(
                f"initial_labeled={cfg['initial_labeled']} must be smaller than train size={len(train_ids)}"
            )
        oracle_ids = select_initial_labeled(
            train_ids=train_ids,
            n_init=cfg["initial_labeled"],
            seed=cfg["seed"],
        )
        residual_ids = sorted_ids([sid for sid in train_ids if sid not in set(oracle_ids)])
        source = "deterministic_from_seed"

    train_id_set = set(train_ids)
    unknown_ids = [sid for sid in (oracle_ids + residual_ids) if sid not in train_id_set]
    if unknown_ids:
        raise ValueError(f"Oracle/residual ids contain unknown train ids: {unknown_ids[:5]}")

    overlap = set(oracle_ids) & set(residual_ids)
    if overlap:
        raise ValueError(f"Oracle/residual ids overlap for {len(overlap)} sample(s)")

    return oracle_ids, residual_ids, source


# ─── Main run ────────────────────────────────────────────────────────────────

def run_finetune(
    cfg: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
) -> None:
    set_seed(cfg["seed"])
    device = setup_device(cfg["gpu"])

    splits, num_landmarks = load_dataset_splits(cfg)
    train_records = sorted_records_by_id(splits["train"])
    val_records = sorted_records_by_id(splits["val"])
    test_records = sorted_records_by_id(splits["test"])
    train_by_id = {r["sample_id"]: r for r in train_records}
    train_ids = [r["sample_id"] for r in train_records]

    base_dir = os.path.join(
        cfg["output_dir"], cfg["dataset"], f"seed_{cfg['seed']}",
        f"active_{cfg['initial_labeled']}", "base",
    )
    default_checkpoint = os.path.join(base_dir, "base_checkpoint.pt")
    cfg_checkpoint = cfg.get("finetune_checkpoint_path")
    checkpoint_candidate = checkpoint_path or cfg_checkpoint
    source_checkpoint = os.path.abspath(checkpoint_candidate) if checkpoint_candidate else default_checkpoint
    if not os.path.exists(source_checkpoint):
        if checkpoint_path:
            raise FileNotFoundError(f"Checkpoint not found: {source_checkpoint}")
        raise FileNotFoundError(
            f"Missing base checkpoint '{source_checkpoint}'. "
            "Run train_base.py first or pass --checkpoint."
        )

    oracle_ids, residual_ids, oracle_ids_source = _resolve_oracle_and_residual_ids(
        cfg=cfg,
        train_ids=train_ids,
        base_dir=base_dir,
    )

    oracle_records = subset_records(train_by_id, oracle_ids)
    residual_records = subset_records(train_by_id, residual_ids)

    gamma = 1.0 if cfg.get("schedule_gamma") is None else float(cfg["schedule_gamma"])
    percentiles, boundaries = build_percentile_schedule(
        num_stages=cfg["num_stages"], percentile_cap=cfg["percentile_cap"],
        total_epochs=cfg["total_epochs"], gamma=gamma,
    )
    tag = schedule_tag(cfg["num_stages"], cfg["percentile_cap"], gamma)

    # Admission mode (also validated in config.py). ``gating`` selects image-level
    # (default PACER) vs per-landmark percentile gating; ``admission_random``
    # swaps the lowest-ERE ranking for a uniformly random subset of the same
    # per-stage size (selection-signal control). The two are mutually exclusive.
    gating = str(cfg.get("gating") or "image").strip().lower()
    if gating not in {"image", "landmark"}:
        raise ValueError(f"gating must be 'image' or 'landmark', got '{cfg.get('gating')}'")
    admission_random = bool(cfg.get("admission_random") or False)
    if gating == "landmark" and admission_random:
        raise ValueError("admission_random is only supported with gating='image'")
    variant = "per_landmark" if gating == "landmark" else ("random" if admission_random else "percentile")

    run_dir = _run_dir(cfg, tag, variant)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    save_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "run_dir": run_dir,
            "active_samples": int(cfg["initial_labeled"]),
            "schedule_tag": tag,
            "stage_percentiles": percentiles,
            "stage_epoch_boundaries": boundaries,
            "checkpoint_path": source_checkpoint,
            "oracle_ids_source": oracle_ids_source,
            **cfg,
            "gating": gating,
            "admission_random": admission_random,
            "schedule_gamma": gamma,
        },
    )

    model = create_landmark_model(cfg=cfg, num_landmarks=num_landmarks, device=device)
    ckpt = torch.load(source_checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["finetune_learning_rate"],
        weight_decay=cfg["finetune_weight_decay"],
    )
    loss_fn = CustomNLLLoss()
    val_loader = create_dataloader(records=val_records, cfg=cfg, shuffle=False, seed=cfg["seed"])

    history: List[Dict[str, Any]] = []
    refresh_log: List[Dict[str, Any]] = []

    final_checkpoint_path = os.path.join(run_dir, "final_checkpoint.pt")

    global_epoch = 0
    for stage_idx, percentile in enumerate(percentiles):
        end_epoch = boundaries[stage_idx]
        start_epoch = boundaries[stage_idx - 1] if stage_idx > 0 else 0
        stage_epochs = end_epoch - start_epoch

        tqdm.write(
            f"=== stage {stage_idx} percentile={percentile:.1f}% "
            f"epochs {start_epoch + 1}..{end_epoch} ==="
        )

        score_table = score_pool(
            model=model, records=residual_records, cfg=cfg, device=device,
        )
        if gating == "landmark":
            landmark_masks = build_per_landmark_percentile_masks(
                score_table, percentile=float(percentile)
            )
            pseudo_labels = build_pseudo_labels_per_landmark(score_table, landmark_masks)
            selected_ids = sorted(pseudo_labels.keys())
        elif admission_random:
            # Draw seed as in the runs behind the paper's selection-signal tables.
            selected_ids = select_random_by_percentile(
                score_table, percentile=float(percentile), seed=cfg["seed"] * 1000 + stage_idx,
            )
            pseudo_labels = build_pseudo_labels(score_table, selected_ids)
        else:
            selected_ids = select_by_percentile(score_table, percentile=float(percentile))
            pseudo_labels = build_pseudo_labels(score_table, selected_ids)

        refresh_log.append(
            _stage_entry(
                stage_idx=stage_idx, percentile=float(percentile),
                start_epoch=start_epoch + 1, end_epoch=end_epoch,
                score_table=score_table, selected_ids=selected_ids,
                previous=refresh_log,
            )
        )

        if selected_ids:
            train_records_this_stage = oracle_records + subset_records(train_by_id, selected_ids)
        else:
            print(
                f"[WARNING] Stage {stage_idx} (percentile={percentile:.1f}%): "
                f"pseudo-pool empty. Training on oracle pool only ({len(oracle_records)}).",
                file=sys.stderr,
            )
            train_records_this_stage = list(oracle_records)

        train_records_this_stage = sorted_records_by_id(train_records_this_stage)
        train_loader = create_dataloader(
            records=train_records_this_stage, cfg=cfg,
            shuffle=True, seed=cfg["seed"],
            pseudo_labels=pseudo_labels if pseudo_labels else None,
        )

        for _ in range(stage_epochs):
            global_epoch += 1
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate_one_epoch(model, val_loader, loss_fn, device)

            history.append({
                "epoch": int(global_epoch),
                "stage_idx": int(stage_idx),
                "percentile": float(percentile),
                "pool_size": int(len(selected_ids)),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_mre": float(val_metrics["mre"]),
                "val_ere": float(val_metrics["ere"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "oracle_count": int(len(oracle_ids)),
            })

            tqdm.write(
                f"  epoch={global_epoch:02d} stage={stage_idx} pct={percentile:.1f}% "
                f"pool={len(selected_ids):03d} "
                f"train={train_loss:.4f} val={val_metrics['loss']:.4f} "
                f"mre={val_metrics['mre']:.3f} ere={val_metrics['ere']:.3f}"
            )

    torch.save({"model_state_dict": model.state_dict(), "epoch": global_epoch}, final_checkpoint_path)
    save_json(os.path.join(run_dir, "training_history.json"), history)
    save_json(os.path.join(run_dir, "pseudo_refresh_log.json"), refresh_log)

    plot_training_curves(
        history={
            "train_loss": [row["train_loss"] for row in history],
            "val_loss": [row["val_loss"] for row in history],
        },
        save_path=os.path.join(plots_dir, "training_curves.png"),
        title=f"{tag} ({cfg['dataset']})",
    )

    _plot_metric_curve(
        epochs=[row["epoch"] for row in history],
        values=[row["val_mre"] for row in history],
        boundaries=boundaries, label="val MRE",
        save_path=os.path.join(plots_dir, "val_mre_by_epoch.png"),
    )
    _plot_metric_curve(
        epochs=[row["epoch"] for row in history],
        values=[row["val_ere"] for row in history],
        boundaries=boundaries, label="val ERE",
        save_path=os.path.join(plots_dir, "val_ere_by_epoch.png"),
    )

    val_res = evaluate_model(
        model=model, records=val_records, cfg=cfg, device=device,
        num_landmarks=num_landmarks, seed=cfg["seed"],
        save_dir=os.path.join(run_dir, "eval_final"),
        split_name="val", grid_samples=cfg["grid_samples"],
    )
    test_res = evaluate_model(
        model=model, records=test_records, cfg=cfg, device=device,
        num_landmarks=num_landmarks, seed=cfg["seed"],
        save_dir=os.path.join(run_dir, "eval_final"),
        split_name="test", grid_samples=cfg["grid_samples"],
    )
    final_metrics = {"val": strip_metrics(val_res), "test": strip_metrics(test_res)}
    save_json(os.path.join(run_dir, "metrics_final.json"), final_metrics)

    _plot_comparison_bars_mre_ere(run_dir, base_dir, final_metrics)

    _write_report(
        run_dir=run_dir, tag=tag, cfg=cfg,
        history=history, refresh_log=refresh_log,
        final_metrics=final_metrics,
        source_checkpoint=source_checkpoint,
        oracle_ids_source=oracle_ids_source,
    )

    print(
        f"[finetune] {tag} done. "
        f"run_dir={run_dir} "
        f"final_test_MRE={final_metrics['test']['MRE']:.3f}"
    )


def _plot_comparison_bars_mre_ere(
    run_dir: str, base_dir: str, final_metrics: Dict[str, Any],
) -> None:
    """Comparative 2x2 bar chart (MRE, ERE, P95, SDR@2mm) of base vs finetuned."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_metrics_path = os.path.join(base_dir, "base_metrics.json")
    if not os.path.exists(base_metrics_path):
        return

    try:
        with open(base_metrics_path) as f:
            base_metrics = json.load(f)
    except Exception as e:
        print(f"[Warning] Could not load base metrics: {e}")
        return

    base_test = base_metrics.get("test", {})
    finetuned_test = final_metrics.get("test", {})

    base_mre = float(base_test.get("MRE", 0))
    base_ere = float(base_test.get("ERE", 0))
    base_p95 = float(base_test.get("P95", 0))
    base_sdr = float(base_test.get("SDR", {}).get("2.0mm", 0))

    finetune_mre = float(finetuned_test.get("MRE", 0))
    finetune_ere = float(finetuned_test.get("ERE", 0))
    finetune_p95 = float(finetuned_test.get("P95", 0))
    finetune_sdr = float(finetuned_test.get("SDR", {}).get("2.0mm", 0))

    if base_mre == 0 and base_ere == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_pos = np.arange(2)

    panels = [
        (axes[0, 0], "MRE (mm)", "Mean Radial Error (MRE) - Lower is Better",
         [base_mre, finetune_mre], ["#1f77b4", "#ff7f0e"], "{:.3f}", None),
        (axes[0, 1], "ERE (mm)", "Expected Radial Error (ERE) - Lower is Better",
         [base_ere, finetune_ere], ["#2ca02c", "#d62728"], "{:.3f}", None),
        (axes[1, 0], "P95 (mm)", "95th Percentile Error (P95) - Lower is Better",
         [base_p95, finetune_p95], ["#9467bd", "#8c564b"], "{:.3f}", None),
        (axes[1, 1], "SDR (%)", "Success Detection Rate @ 2mm - Higher is Better",
         [base_sdr, finetune_sdr], ["#e377c2", "#7f7f7f"], "{:.1f}%", (0, 100)),
    ]

    for ax, ylabel, title, values, colors, fmt, ylim in panels:
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Base", "Finetuned"], fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if ylim is not None:
            ax.set_ylim(*ylim)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Test Set Metrics: Base vs Finetuned", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()

    save_path = os.path.join(run_dir, "plots", "comparison_base_vs_finetuned.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[finetune] saved comparison chart: {save_path}")


def _plot_metric_curve(
    epochs: List[int], values: List[float], boundaries: List[int],
    label: str, save_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, values, "b-", lw=2, label=label)
    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", ls=":", lw=0.8)
    ax.set(xlabel="Epoch", ylabel=label, title=label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    run_dir: str, tag: str, cfg: Dict[str, Any],
    history: List[Dict[str, Any]], refresh_log: List[Dict[str, Any]],
    final_metrics: Dict[str, Any],
    source_checkpoint: str,
    oracle_ids_source: str,
) -> None:
    lines = [
        "=" * 78,
        f"Run: {tag}  dataset={cfg['dataset']}  seed={cfg['seed']}",
        "=" * 78,
        f"Run dir: {run_dir}",
        f"Active samples: {cfg['initial_labeled']}",
        f"Source checkpoint: {source_checkpoint}",
        f"Oracle ids source: {oracle_ids_source}",
        "",
        "Curriculum summary:",
    ]
    for entry in refresh_log:
        mean_ere = entry.get("mean_selected_sample_ere")
        ere_txt = "NA" if mean_ere is None else f"{mean_ere:.3f}"
        eff_thr = entry.get("effective_threshold_mm")
        eff_txt = "NA" if eff_thr is None else f"{eff_thr:.3f}"
        jac = entry.get("jaccard_with_previous")
        jac_txt = "NA" if jac is None else f"{jac:.2f}"
        lines.append(
            f"  stage={entry['stage_idx']} pct={entry['percentile']:.1f}% "
            f"epochs={entry['start_epoch']}..{entry['end_epoch']} "
            f"selected={entry['n_selected_samples']}/{entry['n_total_residual']} "
            f"frac={entry['fraction_selected']:.2f} "
            f"jaccard_vs_prev={jac_txt} "
            f"mean_sample_ere={ere_txt} effective_thr={eff_txt}mm"
        )
    lines.append("")
    lines.append(
        f"Final checkpoint: val_MRE={final_metrics['val']['MRE']:.3f} "
        f"test_MRE={final_metrics['test']['MRE']:.3f} test_ERE={final_metrics['test']['ERE']:.3f}"
    )
    with open(os.path.join(run_dir, "report.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Percentile-gated curriculum finetuner")
    parser.add_argument("--config", default="pacer/configs/finetune.json", help="Path to finetune config JSON")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to finetune from. Overrides config and default base checkpoint.",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, mode="finetune")
    apply_overrides(cfg, args.override)
    run_finetune(cfg, checkpoint_path=args.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
