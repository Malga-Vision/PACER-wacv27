"""Main Fig. "fig:admission5" (--budget 5) and Suppl. Fig. "fig:admission" (--budget 10).

Writes self_tightening_qual_full_{budget}shot.pdf: one row per dataset (DHA,
ISBI2015, Shenzhen), one column per curriculum stage. Each panel shows one pool
image with its predicted heatmap under the model that scored that stage, a
boxed zoom on one tracked landmark, and the image ERE / true MRE in the title.

Needs a 4-stage PACER run (``pacer4``) of the given seed trained with
``--override dump_stage_artifacts=true``, so that its run dir holds
stage_artifacts/stage{t}_model.pt and stage{t}_scores.json for t = 0..3.

Selection (per dataset, over the unlabeled pool):
  * converged = final-stage image MRE within the lowest 40% of the pool;
  * a landmark qualifies if its ERE and MRE both decrease at every stage;
  * each qualifying landmark is ranked by its stage-1 prominence in ERE + MRE
    (each normalized by the image's max), and the image+landmark maximizing it
    is shown, preferring images whose image ERE and MRE also fall every stage
    (filters relaxed in order if no image passes).
"""

import argparse
import json
import os

ROWS = ["hand", "cephalo", "chest"]  # row order of the paper figure
N_STAGES = 4
W = 24  # half-size of the zoom window, pixels


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="root of the run tree")
    ap.add_argument("--dataset-root", required=True, help="root of the landmark datasets")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--budget", type=int, choices=[5, 10], required=True)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--device", default=None, help="torch device (default: cuda:0 if available)")
    args = ap.parse_args()

    import gc
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import ConnectionPatch, Rectangle

    from common import DATASETS, eval_cfg, run_dir
    from shared.data import create_dataloader, load_dataset_splits
    from shared.metrics import CustomNLLLoss, summarize_sample_errors
    from shared.model import create_landmark_model

    # transparent -> red -> yellow, so the image shows through where there is no mass
    cmap_y = LinearSegmentedColormap.from_list(
        "hot_alpha",
        [(0.0, (1, 0, 0, 0.0)), (0.35, (1, 0, 0, 0.85)), (0.7, (1, 0.55, 0, 0.95)), (1.0, (1, 1, 0, 1.0))],
        N=256)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    meta = {ds: (title, unit) for ds, title, unit in DATASETS}

    def load(cfg, k, path):
        m = create_landmark_model(cfg, k, device)
        ck = torch.load(path, map_location=device, weights_only=True)
        m.load_state_dict(ck.get("model_state_dict", ck))
        m.eval()
        return m

    def run(cfg, model, by_id, sids, keep_maps):
        """Per-sample errors; heatmaps/images kept only when keep_maps (chosen image)."""
        loader = create_dataloader(records=[by_id[s] for s in sids], cfg=cfg, shuffle=False, seed=args.seed)
        out = {}
        with torch.no_grad():
            for batch in loader:
                probs = CustomNLLLoss.two_d_softmax(model(batch["image"].to(device))).cpu().numpy()
                scal = batch["physical_scaling_factor"].cpu().numpy()
                gts = batch["landmarks"].cpu().numpy()
                for i, sid in enumerate(batch["name"]):
                    s = summarize_sample_errors(probs[i], gts[i], scal[i])
                    d = dict(lm_ere=np.asarray(s["per_landmark_ere"]), lm_mre=np.asarray(s["per_landmark_mre"]))
                    if keep_maps:
                        d.update(hm=probs[i].copy(), gt=gts[i].copy(), img=batch["image"][i, 0].cpu().numpy())
                    out[str(sid)] = d
        return out

    def per_stage(cfg, k, sa, by_id, sids, keep_maps):
        # one model on the GPU at a time
        res = []
        for t in range(N_STAGES):
            m = load(cfg, k, os.path.join(sa, f"stage{t}_model.pt"))
            res.append(run(cfg, m, by_id, sids, keep_maps))
            del m
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return res

    rows = []  # (title, unit, per-stage dicts of the chosen image, tracked landmark)
    for ds in ROWS:
        title, unit = meta[ds]
        cfg = eval_cfg(args.results, ds, args.seed, args.budget, args.dataset_root, batch_size=8)
        sa = os.path.join(run_dir(args.results, ds, args.seed, args.budget, "pacer4"), "stage_artifacts")
        splits, k = load_dataset_splits(cfg)
        by_id = {r["sample_id"]: r for r in splits["train"]}
        with open(os.path.join(sa, "stage0_scores.json")) as fh:
            pool_ids = list(json.load(fh)["scores"].keys())

        M = per_stage(cfg, k, sa, by_id, pool_ids, keep_maps=False)
        lm_ere = np.array([[M[t][s]["lm_ere"] for t in range(N_STAGES)] for s in pool_ids])  # [N,T,K]
        lm_mre = np.array([[M[t][s]["lm_mre"] for t in range(N_STAGES)] for s in pool_ids])  # [N,T,K]
        img_ere, img_mre = lm_ere.mean(2), lm_mre.mean(2)                                   # [N,T]

        img_viol = (np.sum(np.diff(img_ere, axis=1) > 1e-6, axis=1)
                    + np.sum(np.diff(img_mre, axis=1) > 1e-6, axis=1))
        conv = img_mre[:, -1] <= np.percentile(img_mre[:, -1], 40)
        lm_mono = (np.all(np.diff(lm_mre, axis=1) <= 1e-6, axis=1)
                   & np.all(np.diff(lm_ere, axis=1) <= 1e-6, axis=1))                     # [N,K]
        has_qual = lm_mono.any(axis=1)
        e0, m0 = lm_ere[:, 0, :], lm_mre[:, 0, :]
        prom = np.where(lm_mono,
                        e0 / (e0.max(axis=1, keepdims=True) + 1e-9)
                        + m0 / (m0.max(axis=1, keepdims=True) + 1e-9), -np.inf)
        best_lm, best_sc = prom.argmax(axis=1), prom.max(axis=1)
        for filt in ((img_viol == 0) & conv & has_qual, (img_viol == 0) & has_qual,
                     conv & has_qual, has_qual, np.ones(len(pool_ids), bool)):
            cidx = np.where(filt)[0]
            if len(cidx):
                break
        rep_idx = int(cidx[np.argmax(best_sc[cidx])])
        rep = pool_ids[rep_idx]
        kk = int(best_lm[rep_idx]) if np.isfinite(best_sc[rep_idx]) else int(np.argmax(lm_ere[rep_idx, 0]))
        det = [d[rep] for d in per_stage(cfg, k, sa, by_id, [rep], keep_maps=True)]
        rows.append((title, unit, det, kk))
        print(f"{ds}: rep={rep} lm={kk} img_viol={int(img_viol[rep_idx])} "
              f"lm_ERE={lm_ere[rep_idx, :, kk].round(2).tolist()} "
              f"lm_MRE={lm_mre[rep_idx, :, kk].round(2).tolist()} "
              f"img_ERE={img_ere[rep_idx].round(1).tolist()} img_MRE={img_mre[rep_idx].round(1).tolist()}")

    fig, ax = plt.subplots(len(rows), N_STAGES, figsize=(7.0, 5.8))
    for ri, (title, unit, det, kk) in enumerate(rows):
        for t in range(N_STAGES):
            d, axx = det[t], ax[ri, t]
            hm = d["hm"] / np.max(d["hm"], axis=(1, 2), keepdims=True)
            axx.imshow(d["img"], cmap="gray")
            axx.imshow(np.max(hm, axis=0), cmap=cmap_y, alpha=0.9)
            axx.scatter(d["gt"][:, 0], d["gt"][:, 1], s=1.6, c="cyan", marker="o", linewidths=0)
            axx.set_title(f"ERE {d['lm_ere'].mean():.1f} / MRE {d['lm_mre'].mean():.1f} {unit}", fontsize=6.6)
            axx.set_xticks([]); axx.set_yticks([])
            # zoom window around the tracked landmark, clamped to the image
            h, w = d["img"].shape
            cy, cx = int(round(d["gt"][kk, 1])), int(round(d["gt"][kk, 0]))
            x0, x1 = max(cx - W, 0), min(cx + W, w - 1)
            y0, y1 = max(cy - W, 0), min(cy + W, h - 1)
            # inset shows only the tracked landmark's heatmap
            iax = axx.inset_axes([0.60, 0.60, 0.40, 0.40])
            iax.imshow(d["img"], cmap="gray")
            iax.imshow(d["hm"][kk] / (d["hm"][kk].max() + 1e-9), cmap=cmap_y, alpha=0.95)
            iax.scatter([d["gt"][kk, 0]], [d["gt"][kk, 1]], s=22, facecolors="none",
                        edgecolors="cyan", linewidths=0.9)
            iax.set_xlim(x0, x1); iax.set_ylim(y1, y0)
            iax.set_xticks([]); iax.set_yticks([])
            for sp in iax.spines.values():
                sp.set_edgecolor("yellow"); sp.set_linewidth(0.9)
            # source box on the full image, dashed connectors to the inset corners
            axx.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="yellow", linewidth=0.9, zorder=4))
            for bx, by, ix, iy in [(x0, y0, 0, 1), (x1, y1, 1, 0)]:
                con = ConnectionPatch(xyA=(bx, by), coordsA=axx.transData, xyB=(ix, iy),
                                      coordsB=iax.transAxes, linestyle=(0, (4, 2)),
                                      linewidth=0.8, color="yellow", zorder=5)
                con.set_clip_on(False)
                axx.add_artist(con)
            if ri == len(rows) - 1:
                axx.set_xlabel(f"stage {t + 1}\n$\\pi\\leq{25 * (t + 1)}$%", fontsize=7.5)
        ax[ri, 0].set_ylabel(title, fontsize=7.5)
    plt.subplots_adjust(hspace=0.30, wspace=0.05)
    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(args.out, f"self_tightening_qual_full_{args.budget}shot.{ext}")
        plt.savefig(p, bbox_inches="tight", dpi=600)
        print("wrote", p)
    plt.close()


if __name__ == "__main__":
    main()
