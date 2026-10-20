"""PACER-HRNet: percentile-ERE-gated curriculum self-training on HRNetV2-W18 + MSE.

Replicates the PACER method (adaptive percentile gate on per-sample mean ERE)
on the HRNet free-heatmap / MSE architecture. Disentangles architecture effects
(U-Net softmax vs HRNet free-heatmap) from the PACER approach.

Output: results-v2/{ds}/seed_{s}/active_{N}/{sched_tag}/pacer_hrnet/
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)

import ssl_methods.common as S
from shared.config import save_json, set_seed, setup_device
from shared.data import load_dataset_splits, sorted_records_by_id, subset_records
from shared.metrics import CustomNLLLoss, calculate_ere
from pacer.selection import select_by_percentile as select_per_sample_percentile

from sota.stld.stld_hrnet import (
    MAP, NET_STRIDE,
    build_net, compute_loss_map, evaluate, EvalDataset, forward_map,
    MapDataset, _loader,
)

METHOD = "pacer_hrnet"

PCT_SCHEDULES = {
    "flat":   [100.0],
    "step-2": [50.0, 100.0],
    "step-4": [25.0, 50.0, 75.0, 100.0],
    "step-8": [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5, 100.0],
}

ORIG_INIT_LR = 1e-4
DEFAULT_WARMUP = 60
DEFAULT_GT_SIGMA = 1.5


@torch.no_grad()
def score_pool_hrnet(net, residual_records, num_lms, bs, workers, seed, device):
    net.eval()
    loader = _loader(EvalDataset(residual_records), bs, False, workers, seed)
    table = {}
    for imgs, _gts, scalings, names in loader:
        imgs = imgs.to(device)
        coords, x1_raw = forward_map(net, imgs)
        probs = CustomNLLLoss.two_d_softmax(x1_raw).cpu().numpy()
        coords = coords.cpu().numpy()
        scalings = scalings.numpy()
        for b in range(coords.shape[0]):
            scale = np.asarray(scalings[b], dtype=np.float64)
            pixel_size = scale * NET_STRIDE
            per_lm_ere = []
            for k in range(num_lms):
                pred_grid = coords[b, k] * MAP
                per_lm_ere.append(float(calculate_ere(probs[b, k], pred_grid * pixel_size, pixel_size)))
            table[str(names[b])] = {
                "per_landmark_ere": per_lm_ere,
                "pred_landmarks": coords[b].astype(np.float64),
            }
    return table


def train_epochs(net, ds, n_epochs, opt, sched, bs, workers, seed, device, desc):
    crit = nn.MSELoss(reduction="sum")
    loader = _loader(ds, bs, True, workers, seed)
    last = 0.0
    for _ in range(n_epochs):
        net.train()
        run, nb = 0.0, 0
        for img, t1, t2, m1, m2 in tqdm(loader, desc=desc, leave=False):
            img, t1, t2 = img.to(device), t1.to(device), t2.to(device)
            m1, m2 = m1.to(device), m2.to(device)
            x1, x2 = net(img)
            loss = compute_loss_map(x1, x2, t1, t2, m1, m2, crit, 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.item())
            nb += 1
        sched.step()
        last = run / max(nb, 1)
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="PACER on HRNetV2-W18 + MSE")
    S.add_common_args(ap)
    ap.add_argument("--schedule_tag", default="step-4", choices=list(PCT_SCHEDULES))
    ap.add_argument("--warmup_epochs", default=DEFAULT_WARMUP, type=int,
                    help="ImageNet->labeled warmup (HRNet analogue of PACER's base ckpt)")
    ap.add_argument("--curriculum_epochs", default=120, type=int,
                    help="total epochs for curriculum stages (excluding warmup)")
    ap.add_argument("--stage_epochs", default=None, type=str,
                    help="JSON list overriding per-stage epoch counts")
    ap.add_argument("--init_lr", default=ORIG_INIT_LR, type=float)
    ap.add_argument("--gt_sigma", default=DEFAULT_GT_SIGMA, type=float)
    ap.add_argument("--os_num", default=1000, type=int,
                    help="oversample the labeled set to this many samples/epoch")
    ap.add_argument("--imagenet_weights", default="", type=str)
    args = ap.parse_args()

    cfg = S.make_cfg(args)
    # PACER-HRNet was trained at batch size 16 (the U-Net pipeline uses 8);
    # honor an explicit --override batch_size if one was passed.
    if not any(o.split("=", 1)[0].strip() == "batch_size" for o in args.override):
        cfg["batch_size"] = 16
    cfg["schedule_tag"] = args.schedule_tag
    set_seed(cfg["seed"])
    device = setup_device(cfg["gpu"])

    splits, num_lms = load_dataset_splits(cfg)
    train_records = sorted_records_by_id(splits["train"])
    val_records = sorted_records_by_id(splits["val"])
    test_records = sorted_records_by_id(splits["test"])
    train_by_id = {r["sample_id"]: r for r in train_records}
    train_ids = [r["sample_id"] for r in train_records]
    base_root = cfg.get("base_root", cfg["output_dir"])
    base_dir = os.path.join(base_root, cfg["dataset"], f"seed_{cfg['seed']}",
                            f"active_{cfg['initial_labeled']}", "base")
    oracle_ids, residual_ids, src = S.resolve_oracle_residual(cfg, train_ids, base_dir)
    oracle_records = subset_records(train_by_id, oracle_ids)
    residual_records = subset_records(train_by_id, residual_ids)

    tag = args.schedule_tag
    percentiles = PCT_SCHEDULES[tag]
    curriculum_epochs = args.curriculum_epochs
    num_stages = len(percentiles)
    boundaries = [round(curriculum_epochs * i / num_stages) for i in range(1, num_stages + 1)]
    boundaries[-1] = curriculum_epochs
    if args.stage_epochs:
        stage_epochs = [int(x) for x in json.loads(args.stage_epochs)]
        assert len(stage_epochs) == len(percentiles), "stage_epochs must match #stages"
    else:
        stage_epochs = [boundaries[0]] + [boundaries[i] - boundaries[i - 1]
                                          for i in range(1, len(boundaries))]
    warmup = int(args.warmup_epochs)
    gt_sigma = float(args.gt_sigma)
    os_num = int(args.os_num)
    lr = float(args.init_lr)
    bs, workers, seed = int(cfg["batch_size"]), int(cfg["num_workers"]), int(cfg["seed"])

    def oversample(recs, n):
        if not recs or n <= len(recs):
            return list(recs)
        reps = (n + len(recs) - 1) // len(recs)
        return (list(recs) * reps)[:n]

    oracle_os = oversample(oracle_records, os_num)

    total_epochs = warmup + sum(stage_epochs)
    cfg["epochs"] = int(total_epochs)
    milestones = [int(round(0.66 * total_epochs)), int(round(0.83 * total_epochs))]

    print(f"[{METHOD}] HRNetV2-W18 schedule={tag} percentiles={percentiles} "
          f"stage_epochs={stage_epochs} warmup={warmup} total_epochs={total_epochs} "
          f"milestones={milestones} gt_sigma={gt_sigma} lr={lr} bs={bs} os_num={os_num} "
          f"num_lms={num_lms} n_oracle={len(oracle_ids)} n_residual={len(residual_ids)} "
          f"imagenet_weights={'set' if args.imagenet_weights else 'NONE(normal-init)'}", flush=True)

    net = build_net(num_lms, args.imagenet_weights, device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    history = []

    if warmup > 0:
        items = [(r, "gt", None) for r in oracle_os]
        ds_w = MapDataset(items, num_lms, gt_sigma, None, 0, train=True)
        w_loss = train_epochs(net, ds_w, warmup, opt, sched, bs, workers, seed, device,
                              desc="pacer-hrnet warmup")
        vm = evaluate(net, val_records, num_lms, bs, workers, seed, device)
        history.append({"phase": "warmup", "epochs": warmup, "loss": float(w_loss),
                        "val_MRE": vm["MRE"], "val_ERE": vm["ERE"]})
        tqdm.write(f"  warmup({warmup}ep): loss={w_loss:.4f} val_MRE={vm['MRE']:.3f} val_ERE={vm['ERE']:.3f}")

    for si, (pct, n_ep) in enumerate(zip(percentiles, stage_epochs)):
        score_table = score_pool_hrnet(net, residual_records, num_lms, bs, workers, seed, device)
        admitted = select_per_sample_percentile(score_table, pct)
        admitted_set = set(admitted)
        admitted_eres = [float(np.mean(score_table[sid]["per_landmark_ere"])) for sid in admitted]
        mean_adm_ere = float(np.mean(admitted_eres)) if admitted_eres else None

        items = [(r, "gt", None) for r in oracle_os] + [
            (train_by_id[sid], "pseudo_as_gt", score_table[sid]["pred_landmarks"])
            for sid in residual_ids if sid in admitted_set
        ]
        ds_s = MapDataset(items, num_lms, gt_sigma, None, 0, train=True)
        s_loss = train_epochs(net, ds_s, n_ep, opt, sched, bs, workers, seed + 100 + si, device,
                              desc=f"pacer-hrnet stage{si} p{pct:g}")

        vm = evaluate(net, val_records, num_lms, bs, workers, seed, device)
        history.append({"phase": "curriculum", "stage": si, "percentile": float(pct),
                        "epochs": int(n_ep), "n_admitted": len(admitted),
                        "n_residual": len(residual_ids), "mean_admitted_ere": mean_adm_ere,
                        "loss": float(s_loss), "val_MRE": vm["MRE"], "val_ERE": vm["ERE"]})
        tqdm.write(f"  stage {si} p={pct:g}%: admitted={len(admitted)}/{len(residual_ids)} "
                   f"mean_ere={'-' if mean_adm_ere is None else f'{mean_adm_ere:.3f}'} "
                   f"loss={s_loss:.4f} val_MRE={vm['MRE']:.3f} val_ERE={vm['ERE']:.3f}")

    run_dir = os.path.join(cfg["output_dir"], cfg["dataset"], f"seed_{seed}",
                           f"active_{cfg['initial_labeled']}", tag, METHOD)
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"model_state_dict": net.state_dict()}, os.path.join(run_dir, "final_checkpoint.pt"))
    save_json(os.path.join(run_dir, "training_history.json"), history)
    val_res = evaluate(net, val_records, num_lms, bs, workers, seed, device)
    test_res = evaluate(net, test_records, num_lms, bs, workers, seed, device)
    save_json(os.path.join(run_dir, "metrics_final.json"), {"val": val_res, "test": test_res})
    save_json(os.path.join(run_dir, "run_config.json"), {
        "method": METHOD, "variant": "pacer_percentile_ere_on_hrnet", "backbone": "hrnetv2_w18",
        "schedule_tag": tag, "percentiles": percentiles, "stage_epochs": stage_epochs,
        "warmup_epochs": warmup, "total_epochs": total_epochs, "milestones": milestones,
        "gt_sigma": gt_sigma, "os_num": os_num, "init_lr": lr, "optimizer": "adam_no_wd",
        "net_stride": NET_STRIDE, "selection_free": False, "gate": "per_sample_pct",
        "init": "imagenet", "imagenet_weights": args.imagenet_weights, "oracle_ids_source": src,
        "n_oracle": len(oracle_ids), "n_residual": len(residual_ids), **cfg,
    })
    print(f"[{METHOD}] done. run_dir={run_dir} test_MRE={test_res['MRE']:.3f} test_ERE={test_res['ERE']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())