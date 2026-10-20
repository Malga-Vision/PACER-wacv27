"""FreeMatch: self-adaptive thresholding self-training (no fixed tau).

Faithful adaptation of FreeMatch (Wang et al., ICLR 2023) to dense landmark
regression. Derives thresholds from the model's own confidence stream via two
EMA statistics:

  * global threshold  tau_t   = m * tau_{t-1} + (1-m) * mean(max-confidence over batch)
  * local class conf  p_t(k)  = m * p_{t-1}(k) + (1-m) * mean_b conf[b,k]
  * per-class threshold        tau_t(k) = (p_t(k) / max_j p_t(j)) * tau_t
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
    ap = argparse.ArgumentParser(description="FreeMatch SSL baseline")
    S.add_common_args(ap)
    ap.add_argument("--ema_m", default=0.999, type=float, help="EMA momentum for adaptive thresholds")
    args = ap.parse_args()

    cfg = S.make_cfg(args)
    cfg["epochs"] = int(args.epochs)
    ctx = S.setup(cfg)
    model, device, views = ctx["model"], ctx["device"], ctx["views"]
    K = ctx["num_landmarks"]
    m = float(args.ema_m)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["finetune_learning_rate"],
                            weight_decay=cfg["finetune_weight_decay"])
    loss_fn = CustomNLLLoss(reduction="perkeypoint")
    eval_loss_fn = CustomNLLLoss(reduction="mean")

    tau_g = None
    p_local = None

    lab = S.labeled_cycle(ctx["labeled_loader"])
    rampup = max(1.0, args.rampup_frac * args.epochs)
    history = []

    for epoch in range(args.epochs):
        model.train()
        w = args.lambda_u * S.sigmoid_rampup(epoch, rampup)
        run_sup = run_uns = 0.0
        n_admit = n_tot = 0
        for ub in tqdm(ctx["unlabeled_loader"], desc=f"FreeMatch ep{epoch+1}", leave=False):
            lb = next(lab)
            sup = S.supervised_loss(model, lb, loss_fn, device)

            canon = ub["canon"].numpy()
            B = canon.shape[0]
            weak = torch.stack([views.weak_tensor(canon[b]) for b in range(B)]).to(device)
            with torch.no_grad():
                probs_w = S.two_d_softmax(model(weak)).cpu().numpy()
            coords, conf = S.decode_probs(probs_w)

            if tau_g is None:
                tau_g = float(np.mean(conf))
                p_local = np.mean(conf, axis=0)
            else:
                tau_g = m * tau_g + (1.0 - m) * float(np.mean(conf))
                p_local = m * p_local + (1.0 - m) * np.mean(conf, axis=0)

            p_max = float(np.max(p_local)) if p_local.max() > 0 else 1.0
            tau_k = (p_local / p_max) * tau_g
            mask = (conf >= tau_k[None, :]).astype(np.float32)

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
            "admit_rate": adm, "tau_g": tau_g,
            "val_loss": val["loss"], "val_mre": val["mre"], "val_ere": val["ere"],
        })
        tqdm.write(
            f"  ep{epoch+1:03d} sup={history[-1]['sup_loss']:.4f} "
            f"uns={history[-1]['unsup_loss']:.4f} w={w:.3f} "
            f"admit={adm:.1f}% tau_g={tau_g:.5f} val_mre={val['mre']:.3f}"
        )

    ctx["model"] = model
    S.finalize("freematch", cfg, ctx, history, {
        "ema_m": args.ema_m, "rampup_frac": args.rampup_frac, "init": "base_checkpoint",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())