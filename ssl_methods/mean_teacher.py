"""Mean Teacher: EMA-teacher consistency self-training for landmark heatmaps.

Faithful adaptation of Mean Teacher (Tarvainen & Valpola, NeurIPS 2017).
A teacher network is the EMA of the student weights; the student is trained with:

    loss = supervised NLL (oracle) + w(t) * consistency,

where consistency is the MSE between the student's and teacher's predicted
probability maps on unlabeled images under independent intensity perturbations.
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
    ap = argparse.ArgumentParser(description="Mean Teacher SSL baseline")
    S.add_common_args(ap)
    ap.add_argument("--ema_decay", default=0.99, type=float)
    args = ap.parse_args()

    cfg = S.make_cfg(args)
    cfg["epochs"] = int(args.epochs)
    ctx = S.setup(cfg)
    student, device, views = ctx["model"], ctx["device"], ctx["views"]
    teacher = S.clone_model(student)

    opt = torch.optim.AdamW(student.parameters(), lr=cfg["finetune_learning_rate"],
                            weight_decay=cfg["finetune_weight_decay"])
    loss_fn = CustomNLLLoss(reduction="perkeypoint")
    eval_loss_fn = CustomNLLLoss(reduction="mean")

    lab = S.labeled_cycle(ctx["labeled_loader"])
    rampup = max(1.0, args.rampup_frac * args.epochs)
    history = []

    for epoch in range(args.epochs):
        student.train()
        w = args.lambda_u * S.sigmoid_rampup(epoch, rampup)
        run_sup = run_cons = 0.0
        for ub in tqdm(ctx["unlabeled_loader"], desc=f"MeanTeacher ep{epoch+1}", leave=False):
            lb = next(lab)
            sup = S.supervised_loss(student, lb, loss_fn, device)

            canon = ub["canon"].numpy()
            B = canon.shape[0]
            student_view = torch.stack([views.intensity_tensor(canon[b]) for b in range(B)]).to(device)
            teacher_view = torch.stack([views.intensity_tensor(canon[b]) for b in range(B)]).to(device)

            student.eval()
            with torch.no_grad():
                t_probs = S.two_d_softmax(teacher(teacher_view))
            s_probs = S.two_d_softmax(student(student_view))
            cons = torch.mean(torch.sum((s_probs - t_probs) ** 2, dim=(2, 3)))

            student.train()
            loss = sup + w * cons
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            S.ema_update(teacher, student, args.ema_decay)
            run_sup += float(sup.item())
            run_cons += float(cons.item())

        val = evaluate_one_epoch(teacher, ctx["val_loader"], eval_loss_fn, device)
        history.append({
            "epoch": epoch + 1, "sup_loss": run_sup / max(len(ctx["unlabeled_loader"]), 1),
            "consistency_loss": run_cons / max(len(ctx["unlabeled_loader"]), 1), "w": w,
            "val_loss": val["loss"], "val_mre": val["mre"], "val_ere": val["ere"],
        })
        tqdm.write(
            f"  ep{epoch+1:03d} sup={history[-1]['sup_loss']:.4f} "
            f"cons={history[-1]['consistency_loss']:.4f} w={w:.3f} "
            f"val_mre={val['mre']:.3f}"
        )

    ctx["model"] = teacher
    S.finalize("mean_teacher", cfg, ctx, history, {
        "ema_decay": args.ema_decay, "lambda_u": args.lambda_u,
        "eval_model": "teacher", "init": "base_checkpoint",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())