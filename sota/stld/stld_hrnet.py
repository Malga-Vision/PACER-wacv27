"""STLD-HRNet: HRNetV2-W18 + map head + MSE, selection-free self-training.

Faithful port of STLD (Jin, Che & Chen, arXiv:2404.04556) under the original
paper settings (HRNetV2-W18 backbone, dual heatmap head, MSE loss, shrink
curriculum). Uses the vendored byte-exact HRNet backbone.

Output: <output_dir>/{ds}/seed_{s}/active_{N}/ssl/stld_hrnet/
"""
import argparse
import copy
import json
import os
import sys

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)

VENDOR = os.path.join(CODE_ROOT, "hrnet", "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

import hrnetv2_config as HV2
from hrnetv2 import HighResolutionNet

import ssl_methods.common as S
from shared.config import make_torch_generator, save_json, seed_worker, set_seed, setup_device
from shared.data import load_dataset_splits, read_image, sorted_records_by_id, subset_records
from shared.metrics import calculate_ere, compute_p95, compute_sdr, compute_scaled_distance

METHOD = "stld_hrnet"
NET_STRIDE = 4
INPUT = 256
MAP = INPUT // NET_STRIDE
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ORIG_GT_SIGMA = 1.5
ORIG_SHRINK_CURRI = [1.5, 2.2, 1.8, 1.5, 1.5]
ORIG_SHRINK_WEIGHT = [1.0, 0.1, 0.1, 1.0, 1.0]
ORIG_SEMI_ITER = 5
ORIG_NUM_EPOCHS = 60
ORIG_DECAY_STEPS = [40, 50]
ORIG_INIT_LR = 1e-4
PSEUDO_START_ROUND = 2


def build_net(num_lms, imagenet_weights, device):
    HV2._C.MODEL.NUM_JOINTS = int(num_lms)
    net = HighResolutionNet(HV2._C)
    net.init_weights(imagenet_weights or "")
    return net.to(device)


def gen_target_map(coords_norm, num_lms, sigma, mh=MAP, mw=MAP):
    tm = np.zeros((num_lms, mh, mw), dtype=np.float32)
    tmp = max(1, int(sigma * 3))
    size = 2 * tmp + 1
    ax = np.arange(size, dtype=np.float32)
    g = np.exp(-((ax - tmp) ** 2 + (ax[:, None] - tmp) ** 2) / (2.0 * sigma * sigma))
    for i in range(num_lms):
        mu_x = min(max(int(coords_norm[i, 0] * mw - 0.5), 0), mw - 1)
        mu_y = min(max(int(coords_norm[i, 1] * mh - 0.5), 0), mh - 1)
        ulx, uly = mu_x - tmp, mu_y - tmp
        brx, bry = mu_x + tmp, mu_y + tmp
        gx0, gy0 = max(0, -ulx), max(0, -uly)
        mx0, my0 = max(0, ulx), max(0, uly)
        mx1, my1 = min(mw, brx + 1), min(mh, bry + 1)
        tm[i, my0:my1, mx0:mx1] = g[gy0:gy0 + (my1 - my0), gx0:gx0 + (mx1 - mx0)]
    return tm


def _resize_pad():
    return [A.LongestMaxSize(max_size=INPUT),
            A.PadIfNeeded(min_height=INPUT, min_width=INPUT, border_mode=cv2.BORDER_CONSTANT)]


def _norm():
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


def _aug():
    return [A.Affine(scale=(0.85, 1.20), translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                     rotate=(-15, 15), shear=(-5, 5), interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.ToGray(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4)]


def _kp(transforms):
    return A.Compose(transforms, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


class MapDataset(Dataset):
    def __init__(self, items, num_lms, gt_sigma, shrink_curri, cur_iter, train):
        self.items = items
        self.num_lms = int(num_lms)
        self.gt_sigma = float(gt_sigma)
        self.shrink_curri = shrink_curri
        self.cur_iter = int(cur_iter)
        self.train = bool(train)
        self._full = _kp(_resize_pad() + (_aug() if train else []) + _norm())
        self._aug_only = _kp((_aug() if train else []) + _norm())
        self._noaug_full = _kp(_resize_pad() + _norm())
        self._noaug_aug = _kp(_norm())
        self._resize_img = A.Compose(_resize_pad())

    def __len__(self):
        return len(self.items)

    def _img(self, record):
        return np.ascontiguousarray(read_image(record["image_path"], channels=3))

    def _transform(self, record, dtype, coords_norm):
        img = self._img(record)
        if dtype == "gt":
            kps = np.asarray(record["landmarks"], dtype=np.float64).tolist()
            out = self._full(image=img, keypoints=kps)
            if len(out["keypoints"]) != self.num_lms:
                out = self._noaug_full(image=img, keypoints=kps)
        else:
            img256 = self._resize_img(image=img)["image"]
            kps = (np.asarray(coords_norm, dtype=np.float64) * INPUT).tolist()
            out = self._aug_only(image=img256, keypoints=kps)
            if len(out["keypoints"]) != self.num_lms:
                out = self._noaug_aug(image=img256, keypoints=kps)
        kps_t = np.asarray(out["keypoints"], dtype=np.float64)
        kps_t[:, 0] = np.clip(kps_t[:, 0], 0, INPUT - 1)
        kps_t[:, 1] = np.clip(kps_t[:, 1], 0, INPUT - 1)
        return out["image"].float(), kps_t / float(INPUT)

    def __getitem__(self, idx):
        record, dtype, coords_norm = self.items[idx]
        img_t, norm = self._transform(record, dtype, coords_norm)
        K = self.num_lms
        t1 = np.zeros((K, MAP, MAP), dtype=np.float32)
        t2 = np.zeros((K, MAP, MAP), dtype=np.float32)
        m1, m2 = np.ones_like(t1), np.ones_like(t2)
        sigma_cur = float(self.shrink_curri[self.cur_iter]) if self.shrink_curri else self.gt_sigma
        if dtype == "pseudo_as_gt":
            m2[:] = 0.0
            t1 = gen_target_map(norm, K, self.gt_sigma)
        elif dtype == "gt":
            m2[:] = 0.0
            sig = sigma_cur if sigma_cur < self.gt_sigma else self.gt_sigma
            t1 = gen_target_map(norm, K, sig)
        else:
            if sigma_cur < self.gt_sigma:
                t1 = gen_target_map(norm, K, sigma_cur)
                m2[:] = 0.0
            else:
                t2 = gen_target_map(norm, K, sigma_cur)
                m1[:] = 0.0
        return (img_t, torch.from_numpy(t1), torch.from_numpy(t2),
                torch.from_numpy(m1), torch.from_numpy(m2))


class EvalDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self._t = _kp(_resize_pad() + _norm())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = np.ascontiguousarray(read_image(r["image_path"], channels=3))
        out = self._t(image=img, keypoints=np.asarray(r["landmarks"]).tolist())
        kps = np.asarray(out["keypoints"], dtype=np.float64)
        return (out["image"].float(), torch.from_numpy(kps),
                torch.from_numpy(np.asarray(r["physical_scaling_factor"], dtype=np.float64)), r["sample_id"])


class UnlabeledDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self._t = A.Compose(_resize_pad() + _norm())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = np.ascontiguousarray(read_image(r["image_path"], channels=3))
        return self._t(image=img)["image"].float(), r["sample_id"]


def _loader(ds, bs, shuffle, workers, seed):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      drop_last=False, worker_init_fn=seed_worker, generator=make_torch_generator(seed))


def compute_loss_map(x1, x2, t1, t2, m1, m2, crit, shrink_weight):
    loss1 = crit(x1 * m1, t1 * m1)
    if m1.sum() != 0:
        loss1 = loss1 / m1.sum()
    loss2 = crit(x2 * m2, t2 * m2)
    if m2.sum() != 0:
        loss2 = loss2 / m2.sum()
    return loss1 + shrink_weight * loss2


@torch.no_grad()
def forward_map(net, inputs):
    net.eval()
    outputs, _ = net(inputs)
    B, K, H, W = outputs.shape
    flat = outputs.reshape(B * K, -1)
    max_ids = torch.argmax(flat, 1).view(-1, 1)
    pad_x, pad_y = nn.ZeroPad2d((1, 1, 0, 0)), nn.ZeroPad2d((0, 0, 1, 1))
    shx = pad_x(outputs[:, :, :, 2:] - outputs[:, :, :, :-2]).reshape(B * K, -1)
    shy = pad_y(outputs[:, :, 2:, :] - outputs[:, :, :-2, :]).reshape(B * K, -1)
    sx = torch.gather(shx, 1, max_ids).squeeze(1).sign() * 0.25
    sy = torch.gather(shy, 1, max_ids).squeeze(1).sign() * 0.25
    tx = ((max_ids.squeeze(1) % W).float() + 1 + sx) / (INPUT / NET_STRIDE)
    ty = (torch.div(max_ids.squeeze(1), W, rounding_mode="trunc").float() + 1 + sy) / (INPUT / NET_STRIDE)
    return torch.stack([tx, ty], 1).view(B, K, 2), outputs


@torch.no_grad()
def gen_pseudo(net, residual_records, bs, workers, seed, device):
    loader = _loader(UnlabeledDataset(residual_records), bs, False, workers, seed)
    pseudo = {}
    for imgs, names in loader:
        coords, _ = forward_map(net, imgs.to(device))
        coords = coords.cpu().numpy()
        for b in range(coords.shape[0]):
            pseudo[str(names[b])] = coords[b]
    return pseudo


def train_stage(net, ds, epochs, milestones, lr, shrink_weight, bs, workers, seed, device, desc):
    crit = nn.MSELoss(reduction="sum")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    loader = _loader(ds, bs, True, workers, seed)
    loss_sum, n_ep = 0.0, 0
    for _ in range(epochs):
        net.train()
        run, nb = 0.0, 0
        for img, t1, t2, m1, m2 in tqdm(loader, desc=desc, leave=False):
            img, t1, t2 = img.to(device), t1.to(device), t2.to(device)
            m1, m2 = m1.to(device), m2.to(device)
            x1, x2 = net(img)
            loss = compute_loss_map(x1, x2, t1, t2, m1, m2, crit, shrink_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.item())
            nb += 1
        sched.step()
        loss_sum += run / max(nb, 1)
        n_ep += 1
    return loss_sum / max(n_ep, 1)


@torch.no_grad()
def evaluate(net, records, num_lms, bs, workers, seed, device):
    loader = _loader(EvalDataset(records), bs, False, workers, seed)
    per_lm_mre, per_lm_ere = [], []
    for imgs, gts, scalings, _names in loader:
        coords, x1 = forward_map(net, imgs.to(device))
        coords, x1 = coords.cpu().numpy(), x1.cpu().numpy()
        gts, scalings = gts.numpy(), scalings.numpy()
        for b in range(coords.shape[0]):
            scale = np.asarray(scalings[b], dtype=np.float64)
            pixel_size = scale * NET_STRIDE
            for k in range(num_lms):
                pred_px = coords[b, k] * INPUT
                per_lm_mre.append(compute_scaled_distance(pred_px, gts[b, k], scale))
                pred_grid = coords[b, k] * MAP
                per_lm_ere.append(float(calculate_ere(x1[b, k], pred_grid * pixel_size, pixel_size)))
    re_arr = np.asarray(per_lm_mre, dtype=np.float64)
    ere_arr = np.asarray(per_lm_ere, dtype=np.float64)
    return {
        "MRE": float(re_arr.mean()) if re_arr.size else 0.0,
        "ERE": float(ere_arr.mean()) if ere_arr.size else 0.0,
        "P95": compute_p95(re_arr), "loss": 0.0,
        "SDR": compute_sdr(re_arr),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="STLD-HRNet (HRNetV2-W18 + map + MSE)")
    S.add_common_args(ap)
    ap.add_argument("--rounds", default=ORIG_SEMI_ITER, type=int)
    ap.add_argument("--num_epochs", default=ORIG_NUM_EPOCHS, type=int)
    ap.add_argument("--pretrain_speedup", default=5, type=int)
    ap.add_argument("--init_lr", default=ORIG_INIT_LR, type=float)
    ap.add_argument("--decay_steps", default=None, type=str)
    ap.add_argument("--gt_sigma", default=ORIG_GT_SIGMA, type=float)
    ap.add_argument("--shrink_curri", default=None, type=str)
    ap.add_argument("--shrink_weight_curri", default=None, type=str)
    ap.add_argument("--pseudo_start_round", default=PSEUDO_START_ROUND, type=int)
    ap.add_argument("--os_num", default=1000, type=int,
                    help="oversample the labeled set to this many samples/epoch")
    ap.add_argument("--imagenet_weights", default="", type=str,
                    help="path to hrnetv2_w18_imagenet_pretrained.pth (else normal-init)")
    args = ap.parse_args()

    cfg = S.make_cfg(args)
    # STLD-HRNet was trained at batch size 16 (the U-Net pipeline uses 8);
    # honor an explicit --override batch_size if one was passed.
    if not any(o.split("=", 1)[0].strip() == "batch_size" for o in args.override):
        cfg["batch_size"] = 16
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

    T = int(args.rounds)
    num_epochs = int(args.num_epochs)
    speedup = max(1, int(args.pretrain_speedup))
    pp_later = max(1, int(round(num_epochs / speedup)))
    lr = float(args.init_lr)
    decay_steps = [int(x) for x in (json.loads(args.decay_steps) if args.decay_steps else ORIG_DECAY_STEPS)]
    decay_pp = [max(1, s // speedup) for s in decay_steps]
    gt_sigma = float(args.gt_sigma)
    shrink_curri = [float(x) for x in (json.loads(args.shrink_curri) if args.shrink_curri else ORIG_SHRINK_CURRI)]
    shrink_weight = [float(x) for x in (json.loads(args.shrink_weight_curri) if args.shrink_weight_curri else ORIG_SHRINK_WEIGHT)]
    ps_start = int(args.pseudo_start_round)
    os_num = int(args.os_num)
    bs, workers, seed = int(cfg["batch_size"]), int(cfg["num_workers"]), int(cfg["seed"])

    def oversample(recs, n):
        if not recs or n <= len(recs):
            return list(recs)
        reps = (n + len(recs) - 1) // len(recs)
        return (list(recs) * reps)[:n]

    oracle_os = oversample(oracle_records, os_num)

    pp_per_round = [0] + [(num_epochs if t <= 1 else pp_later) for t in range(1, T)]
    total_epochs = sum(pp_per_round) + T * num_epochs
    cfg["epochs"] = int(total_epochs)

    print(f"[{METHOD}] HRNetV2-W18 T={T} num_epochs={num_epochs} pp_later={pp_later} "
          f"gt_sigma={gt_sigma} shrink_curri={shrink_curri} weight={shrink_weight} ps_start=ti{ps_start} "
          f"lr={lr} bs={bs} num_lms={num_lms} total_epochs={total_epochs} os_num={os_num} "
          f"n_oracle={len(oracle_ids)} n_residual={len(residual_ids)} "
          f"imagenet_weights={'set' if args.imagenet_weights else 'NONE(normal-init)'}", flush=True)

    net_pp = build_net(num_lms, args.imagenet_weights, device)
    pseudo = None
    history = []
    final_net = net_pp

    for t in range(T):
        pp_loss = None
        if pseudo is not None:
            pp_epochs = num_epochs if t <= 1 else pp_later
            pp_ms = decay_steps if t <= 1 else decay_pp
            items = [(r, "pseudo_as_gt", pseudo[r["sample_id"]]) for r in residual_records]
            ds_pp = MapDataset(items, num_lms, gt_sigma, shrink_curri, t, train=True)
            pp_loss = train_stage(net_pp, ds_pp, pp_epochs, pp_ms, lr, 0.0, bs, workers, seed + t, device,
                                  desc=f"STLD-hrnet pp ti{t}")

        # Faithful STLD shrink curriculum: the pseudo consumed this round was
        # GENERATED at round t-1, and the original stamps each pseudo sample with
        # its generating round, so its shrink sigma is shrink_curri[t-1] -> the
        # published sequence [2.2, 1.8, 1.5] over ti=2,3,4. The loss WEIGHT stays
        # indexed by the current round t. For labeled-only rounds (t<ps_start)
        # there is no pseudo, so the index is irrelevant.
        shrink_idx = (t - 1) if t >= ps_start else t
        net_mix = copy.deepcopy(net_pp)
        if t < ps_start:
            lam = 0.0
            items = [(r, "gt", None) for r in oracle_os]
        else:
            lam = float(shrink_weight[t]) if t < len(shrink_weight) else float(shrink_weight[-1])
            items = ([(r, "gt", None) for r in oracle_os]
                     + [(r, "pseudo", pseudo[r["sample_id"]]) for r in residual_records])
        ds_mix = MapDataset(items, num_lms, gt_sigma, shrink_curri, shrink_idx, train=True)
        mix_loss = train_stage(net_mix, ds_mix, num_epochs, decay_steps, lr, lam, bs, workers,
                               seed + 100 + t, device, desc=f"STLD-hrnet mixed ti{t}")

        vm = evaluate(net_mix, val_records, num_lms, bs, workers, seed, device)
        sig_t = float(shrink_curri[shrink_idx]) if shrink_idx < len(shrink_curri) else gt_sigma
        history.append({"round": t, "pp_loss": (None if pp_loss is None else float(pp_loss)),
                        "mix_loss": float(mix_loss), "lambda": float(lam), "shrink_sigma": sig_t,
                        "val_MRE": vm["MRE"], "val_ERE": vm["ERE"]})
        tqdm.write(f"  round {t}/{T-1}: pp={'-' if pp_loss is None else f'{pp_loss:.4f}'} "
                   f"mix={mix_loss:.4f} lam={lam:.2f} sig={sig_t:.2f} val_MRE={vm['MRE']:.3f} val_ERE={vm['ERE']:.3f}")

        final_net = net_mix
        if t < T - 1:
            pseudo = gen_pseudo(net_mix, residual_records, bs, workers, seed, device)

    run_dir = os.path.join(cfg["output_dir"], cfg["dataset"], f"seed_{seed}",
                           f"active_{cfg['initial_labeled']}", "ssl", METHOD)
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"model_state_dict": final_net.state_dict()}, os.path.join(run_dir, "final_checkpoint.pt"))
    save_json(os.path.join(run_dir, "training_history.json"), history)
    val_res = evaluate(final_net, val_records, num_lms, bs, workers, seed, device)
    test_res = evaluate(final_net, test_records, num_lms, bs, workers, seed, device)
    save_json(os.path.join(run_dir, "metrics_final.json"), {"val": val_res, "test": test_res})
    save_json(os.path.join(run_dir, "run_config.json"), {
        "method": METHOD, "variant": "original_architecture_full_port", "backbone": "hrnetv2_w18",
        "rounds": T, "num_epochs": num_epochs, "pretrain_epochs_later": pp_later, "total_epochs": total_epochs,
        "gt_sigma": gt_sigma, "shrink_curri": shrink_curri, "shrink_weight_curri": shrink_weight,
        "pseudo_start_round": ps_start, "os_num": os_num,
        "init_lr": lr, "decay_steps": decay_steps, "optimizer": "adam_no_wd",
        "net_stride": NET_STRIDE, "selection_free": True, "init": "imagenet",
        "imagenet_weights": args.imagenet_weights, "oracle_ids_source": src,
        "n_oracle": len(oracle_ids), "n_residual": len(residual_ids), **cfg,
    })
    print(f"[{METHOD}] done. run_dir={run_dir} test_MRE={test_res['MRE']:.3f} test_ERE={test_res['ERE']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())