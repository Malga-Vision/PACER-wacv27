"""Model factory and optional encoder checkpoint loading.

Only SMP UNet is supported (the same detector as the CDPM-Align downstream landmark task).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _normalize_encoder_weights(encoder_weights: Optional[str]) -> Optional[str]:
    if encoder_weights is None:
        return None
    lowered = encoder_weights.lower().strip()
    if lowered in {"none", "null", "random", ""}:
        return None
    return encoder_weights


def _load_encoder_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if any(key.startswith("backbone.") for key in state_dict):
        state_dict = {
            key[len("backbone."):]: value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }

    missing, unexpected = model.encoder.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading encoder: {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys when loading encoder: {unexpected[:5]}")
    print(f"Loaded encoder weights from: {checkpoint_path}")
    del checkpoint


def create_landmark_model(cfg: Dict[str, Any], num_landmarks: int, device: torch.device) -> nn.Module:
    """Create an SMP UNet for heatmap-based landmark localization."""
    import segmentation_models_pytorch as smp

    arch = str(cfg["architecture"]).lower().replace("smp_", "")
    if arch != "unet":
        raise ValueError(f"Only SMP UNet is supported, got '{cfg['architecture']}'")

    model = smp.Unet(
        encoder_name=cfg["encoder"],
        encoder_weights=_normalize_encoder_weights(cfg["encoder_weights"]),
        in_channels=int(cfg["image_channels"]),
        classes=int(num_landmarks),
        decoder_channels=[256, 128, 64, 32, 16],
    )

    if cfg["pretrained_path"]:
        _load_encoder_checkpoint(model=model, checkpoint_path=cfg["pretrained_path"], device=device)

    return model.to(device)
