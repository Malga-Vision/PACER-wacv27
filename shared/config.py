"""JSON configuration utilities for the percentile-gated self-training pipeline."""

import json
import os
import random
from typing import Any, Dict, List, Set

import numpy as np
import torch


DATASET_NUM_LANDMARKS = {
    "chest": 6,
    "hand": 37,
    "cephalo": 19,
}


COMMON_CONFIG_KEYS = {
    "dataset",
    "dataset_root",
    "output_dir",
    "gpu",
    "seed",
    "architecture",
    "encoder",
    "encoder_weights",
    "pretrained_path",
    "image_size",
    "image_channels",
    "sigma",
    "batch_size",
    "num_workers",
    "pin_memory",
    "initial_labeled",
    "grid_samples",
}


BASE_CONFIG_KEYS = {
    "base_epochs",
    "base_learning_rate",
    "base_weight_decay",
    "base_early_stopping_patience",
}


FINETUNE_CONFIG_KEYS = {
    "finetune_learning_rate",
    "finetune_weight_decay",
    "num_stages",
    "percentile_cap",
    "total_epochs",
}


OPTIONAL_CONFIG_KEYS = {
    "finetune_checkpoint_path",
    "save_ere_extremes_on",
    "gating",
    "admission_random",
    "schedule_gamma",
}


VALID_CONFIG_MODES = {"all", "base", "finetune", "eval"}


def _required_keys_for_mode(mode: str) -> Set[str]:
    if mode == "base":
        return set(COMMON_CONFIG_KEYS | BASE_CONFIG_KEYS)
    if mode == "finetune":
        return set(COMMON_CONFIG_KEYS | FINETUNE_CONFIG_KEYS)
    if mode == "eval":
        return set(COMMON_CONFIG_KEYS)
    return set(COMMON_CONFIG_KEYS | BASE_CONFIG_KEYS | FINETUNE_CONFIG_KEYS)


def _validate_config(cfg: Dict[str, Any], mode: str) -> None:
    if cfg["dataset"] not in DATASET_NUM_LANDMARKS:
        raise ValueError(
            f"Unsupported dataset '{cfg['dataset']}'. Choose from: {list(DATASET_NUM_LANDMARKS)}"
        )

    if cfg["image_size"][0] <= 0 or cfg["image_size"][1] <= 0:
        raise ValueError(f"image_size must be positive, got {cfg['image_size']}")
    if cfg["image_channels"] <= 0:
        raise ValueError(f"image_channels must be >= 1, got {cfg['image_channels']}")
    if cfg["batch_size"] <= 0:
        raise ValueError(f"batch_size must be >= 1, got {cfg['batch_size']}")

    if "base_learning_rate" in cfg:
        if cfg["base_learning_rate"] <= 0:
            raise ValueError(f"base_learning_rate must be > 0, got {cfg['base_learning_rate']}")
        if cfg["base_weight_decay"] < 0:
            raise ValueError(f"base_weight_decay must be >= 0, got {cfg['base_weight_decay']}")
        if cfg["base_early_stopping_patience"] <= 0:
            raise ValueError(
                "base_early_stopping_patience must be >= 1, "
                f"got {cfg['base_early_stopping_patience']}"
            )
        if cfg["base_epochs"] < 1:
            raise ValueError(f"base_epochs must be >= 1, got {cfg['base_epochs']}")

    if "finetune_learning_rate" in cfg:
        if cfg["finetune_learning_rate"] <= 0:
            raise ValueError(
                f"finetune_learning_rate must be > 0, got {cfg['finetune_learning_rate']}"
            )
        if cfg["finetune_weight_decay"] < 0:
            raise ValueError(
                f"finetune_weight_decay must be >= 0, got {cfg['finetune_weight_decay']}"
            )
        if cfg["num_stages"] < 1:
            raise ValueError(f"num_stages must be >= 1, got {cfg['num_stages']}")
        if not 0.0 <= cfg["percentile_cap"] <= 100.0:
            raise ValueError(
                f"percentile_cap must be in [0, 100], got {cfg['percentile_cap']}"
            )
        if cfg["total_epochs"] < cfg["num_stages"]:
            raise ValueError(
                f"total_epochs ({cfg['total_epochs']}) must be >= num_stages ({cfg['num_stages']})"
            )

        # Ablation knobs (optional). gating: image-level (default PACER) vs
        # per-landmark percentile gating; admission_random: random subset of the
        # same per-stage size instead of lowest-ERE (selection-signal control).
        gating = str(cfg.get("gating") or "image").strip().lower()
        if gating not in {"image", "landmark"}:
            raise ValueError(f"gating must be 'image' or 'landmark', got '{cfg.get('gating')}'")
        if gating == "landmark" and bool(cfg.get("admission_random")):
            raise ValueError("admission_random is only supported with gating='image'")
        # Ramp shape: pi_t = cap * (t/T) ** schedule_gamma (null/1 = linear PACER).
        if cfg.get("schedule_gamma") is not None and float(cfg["schedule_gamma"]) <= 0:
            raise ValueError(f"schedule_gamma must be > 0, got {cfg['schedule_gamma']}")

    if cfg["initial_labeled"] < 1:
        raise ValueError(f"initial_labeled must be >= 1, got {cfg['initial_labeled']}")

    if cfg.get("finetune_checkpoint_path") is not None and not isinstance(
        cfg.get("finetune_checkpoint_path"), str
    ):
        raise ValueError("finetune_checkpoint_path must be null or a string path")
    if cfg.get("finetune_checkpoint_path"):
        checkpoint_path = str(cfg["finetune_checkpoint_path"]).strip()
        if checkpoint_path and not os.path.exists(checkpoint_path):
            raise ValueError(f"finetune_checkpoint_path '{checkpoint_path}' does not exist")
    if cfg["grid_samples"] < 0:
        raise ValueError(f"grid_samples must be >= 0, got {cfg['grid_samples']}")


def load_config(path: str, mode: str = "all") -> Dict[str, Any]:
    """Load, coerce, and validate a JSON config dictionary."""
    mode = str(mode).strip().lower()
    if mode not in VALID_CONFIG_MODES:
        raise ValueError(f"Unsupported config mode '{mode}', choose from {sorted(VALID_CONFIG_MODES)}")

    with open(path, "r") as handle:
        payload = json.load(handle)

    payload_keys = {key for key in payload.keys() if not key.startswith("_")}
    required = _required_keys_for_mode(mode)
    if mode == "eval":
        allowed = (
            COMMON_CONFIG_KEYS | BASE_CONFIG_KEYS | FINETUNE_CONFIG_KEYS | OPTIONAL_CONFIG_KEYS
        )
    else:
        allowed = required | OPTIONAL_CONFIG_KEYS

    missing = sorted(required - payload_keys)
    unknown = sorted(payload_keys - allowed)
    if missing:
        raise ValueError(f"Missing required config key(s): {missing}")
    if unknown:
        raise ValueError(f"Unknown config key(s): {unknown}")

    cfg = {key: payload[key] for key in required}
    for key in OPTIONAL_CONFIG_KEYS:
        cfg[key] = payload.get(key)

    cfg["dataset"] = str(cfg["dataset"]).strip().lower()
    cfg["image_size"] = tuple(int(v) for v in cfg["image_size"])
    cfg["gpu"] = int(cfg["gpu"])
    cfg["seed"] = int(cfg["seed"])
    cfg["image_channels"] = int(cfg["image_channels"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["num_workers"] = int(cfg["num_workers"])
    cfg["initial_labeled"] = int(cfg["initial_labeled"])
    cfg["grid_samples"] = int(cfg["grid_samples"])

    if "base_epochs" in cfg:
        cfg["base_epochs"] = int(cfg["base_epochs"])
        cfg["base_learning_rate"] = float(cfg["base_learning_rate"])
        cfg["base_weight_decay"] = float(cfg["base_weight_decay"])
        cfg["base_early_stopping_patience"] = int(cfg["base_early_stopping_patience"])

    if "finetune_learning_rate" in cfg:
        cfg["finetune_learning_rate"] = float(cfg["finetune_learning_rate"])
        cfg["finetune_weight_decay"] = float(cfg["finetune_weight_decay"])
        cfg["num_stages"] = int(cfg["num_stages"])
        cfg["percentile_cap"] = float(cfg["percentile_cap"])
        cfg["total_epochs"] = int(cfg["total_epochs"])

    if cfg.get("finetune_checkpoint_path") is not None:
        cfg["finetune_checkpoint_path"] = str(cfg["finetune_checkpoint_path"]).strip()

    _validate_config(cfg, mode=mode)
    return cfg


def set_seed(seed: int) -> None:
    """Enable deterministic execution for Python, NumPy, and PyTorch.

    Note: PYTHONHASHSEED must be set in the *environment* before the Python
    interpreter starts; setting os.environ here is too late for the current
    process. Launchers set it in the subprocess env. We also set
    CUBLAS_WORKSPACE_CONFIG for deterministic CuBLAS operations under
    torch.use_deterministic_algorithms.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def setup_device(gpu: int) -> torch.device:
    """Resolve the runtime device. Prefers MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif gpu >= 0 and torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if gpu >= cuda_count:
            raise ValueError(f"Requested gpu={gpu} but only {cuda_count} CUDA device(s) available")
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def apply_overrides(cfg: Dict[str, Any], items: List[str]) -> None:
    for item in items:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got '{item}'")
        key, raw = item.split("=", 1)
        key = key.strip()
        try:
            value: Any = json.loads(raw)
        except Exception:
            value = raw
        cfg[key] = value


def save_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
