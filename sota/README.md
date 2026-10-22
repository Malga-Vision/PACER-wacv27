# SOTA comparison methods

Task-specific and pretraining-based competitors compared against PACER.

## `stld/` — STLD-HRNet (bundled)

The shape-aware landmark self-training method of the paper's STLD reference, adapted to our
few-shot protocol on an HRNetV2-W18 backbone (shrinking-σ pseudo-label curriculum, per-pixel
Gaussian-heatmap MSE). Trained from the **same** `N` oracle labels and the **same** unlabeled
pool as PACER-HRNet, so the two differ only in how the pool is mined.

```bash
python main.py stld-hrnet --dataset hand --initial_labeled 5 --seed 42 --gpu 0 \
    --output_dir /path/to/results \
    --imagenet_weights hrnet/vendor/hrnetv2_w18_imagenet_pretrained.pth
```

## `diffusion/` — DDPM & CDPM-Align (external, not bundled)

The diffusion-pretrained few-shot detectors (DDPM and CDPM-Align) learn an anatomy prior
through generative pretraining before any landmark is seen. They are **not** reimplemented
here: CDPM-Align (MICCAI 2026, related work from our lab) has its own official code at
[Malga-Vision/CDPM-Align](https://github.com/Malga-Vision/CDPM-Align); see also
`diffusion/PLACEHOLDER.txt`. The paper's **CDPM-Align + PACER** row runs the PACER curriculum
from a CDPM-Align detector checkpoint (same U-Net/ResNet-101, full state dict) via the
standard finetuner:

```bash
python main.py finetune --config pacer/configs/finetune.json \
    --checkpoint /path/to/cdpm_align_detector_checkpoint.pt \
    --override output_dir=/path/to/results_cdpm_align \
    --override dataset=hand --override initial_labeled=5 --override seed=42
```

The run directory depends only on the schedule tag, so the separate `output_dir` keeps this run
from overwriting the ImageNet-base PACER run.
