"""FlexMatch: curriculum pseudo-labeling on top of FixMatch-pose.

Faithful adaptation of FlexMatch (Zhang et al., NeurIPS 2021) to dense landmark
regression. Replaces the single fixed threshold with per-landmark curriculum
thresholds that reflect each landmark's learning status.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssl_methods import common as S
from shared.metrics import CustomNLLLoss
from shared.train_utils import evaluate_one_epoch


def main() -> int:
    ap = argparse.ArgumentParser(description="FlexMatch SSL baseline")
    S.add_common_args(ap)
    ap.add_argument("--tau_quantile", default=0.8, type=float,
                    help="quantile of oracle peak-confidence for the conservative base tau")
    ap.add_argument("--tau", default=None, type=float)
    args = ap.parse_args()

    cfg = S.make_cfg(args)
    cfg["epochs"] = int(args.epochs)
    ctx = S.setup(cfg)
    model, device, views = ctx["model"], ctx["device"], ctx["views"]
    K = ctx["num_landmarks"]
    U = max(len(ctx["residual_records"]), 1)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["finetune_learning_rate"],
                            weight_decay=cfg["finetune_weight_decay"])
    loss_fn = CustomNLLLoss(reduction="perkeypoint")
    eval_loss_fn = CustomNLLLoss(reduction="mean")

    tau = float(args.tau) if args.tau is not None else S.calibrate_tau(
        model, ctx["oracle_records"], cfg, views, device, args.tau_quantile)
    print(f"[flexmatch] base tau={tau:.5f} U={U} K={K}", flush=True)

    passed_mem = {}

    def class_thresholds() -> np.ndarray:
        if not passed_mem:
            return np.zeros(K, dtype=np.float32)
        sigma = np.sum(np.stack(list(passed_mem.values()), axis=0), axis=0)
        denom = np.maximum(sigma, U - sigma)
        beta = np.clip(sigma / np.maximum(denom, 1.0), 0.0, 1.0)
        return (beta * tau).astype(np.float32)

    lab = S.labeled_cycle(ctx["labeled_loader"])
    rampup = max(1.0, args.rampup_frac * args.epochs)
    history = []

    for epoch in range(args.epochs):
        model.train()
        w = args.lambda_u * S.sigmoid_rampup(epoch, rampup)
        run_sup = run_uns = 0.0
        n_admit = n_tot = 0
        for ub in tqdm(ctx["unlabeled_loader"], desc=f"FlexMatch ep{epoch+1}", leave=False):
            lb = next(lab)
            sup = S.supervised_loss(model, lb, loss_fn, device)

            canon = ub["canon"].numpy()
            B = canon.shape[0]
            weak = torch.stack([views.weak_tensor(canon[b]) for b in range(B)]).to(device)
            with torch.no_grad():
                probs_w = S.two_d_softmax(model(weak)).cpu().numpy()
            coords, conf = S.decode_probs(probs_w)
            # Gate with thresholds from the PRIOR memory (batches 0..t-1), then record
            # this batch's base-threshold pass indicators (FlexMatch CPL, sigma_k).
            tau_k = class_thresholds()
            mask = (conf >= tau_k[None, :]).astype(np.float32)
            base_pass = (conf >= tau).astype(np.float32)
            for b in range(B):
                passed_mem[str(ub["name"][b])] = base_pass[b]

            strong, tgt = [], []
            for b in range(B):
                img_t, lms = views.strong(canon[b], coords[b])
                strong.append(img_t)
                tgt.append(torch.from_numpy(views.heatmap(lms)).float())
            strong = torch.stack(strong).to(device)
            target = torch.stack(tgt).to(device)

            uns_probs = S.two_d_softmax(model(strong))
            per_kp = loss_fn(uns_probs, target)
            uns = S.masked_unsup_loss(per_kp, torch.from_numpy(mask).to(device))
            loss = sup + w * uns

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run_sup += float(sup.item())
            run_uns += float(uns.item())
            n_admit += int(mask.sum())
            n_tot += int(mask.size)


        val = evaluate_one_epoch(model, ctx["val_loader"], eval_loss_fn, device)
        adm = (float(n_admit) / max(n_tot, 1)) * 100.0 if n_tot else 0.0
        history.append({
            "epoch": epoch + 1, "sup_loss": run_sup / max(len(ctx["unlabeled_loader"]), 1),
            "unsup_loss": run_uns / max(len(ctx["unlabeled_loader"]), 1), "w": w,
            "admit_rate": adm, "val_loss": val["loss"], "val_mre": val["mre"], "val_ere": val["ere"],
        })
        tqdm.write(
            f"  ep{epoch+1:03d} sup={history[-1]['sup_loss']:.4f} "
            f"uns={history[-1]['unsup_loss']:.4f} w={w:.3f} "
            f"admit={adm:.1f}% val_mre={val['mre']:.3f}"
        )

    ctx["model"] = model
    S.finalize("flexmatch", cfg, ctx, history, {
        "tau": tau, "tau_quantile": args.tau_quantile,
        "rampup_frac": args.rampup_frac, "init": "base_checkpoint",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())