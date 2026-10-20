"""Dataset parsers, LandmarkDataset, and DataLoader factory for PACER."""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import DATASET_NUM_LANDMARKS, make_torch_generator, seed_worker
from shared.metrics import compute_euclidean_distance, get_scale_factor, keypoints2heatmaps


# ─── Image reading ────────────────────────────────────────────────────────────

def read_image_shape(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return int(height), int(width)


def read_image(image_path: str, channels: int) -> np.ndarray:
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".dcm":
        import pydicom
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(np.float32)
        image = image.max() - image
        if channels == 3:
            return np.stack([image, image, image], axis=-1) if image.ndim == 2 else image
        return np.expand_dims(image, axis=-1) if image.ndim == 2 else image
    if channels == 3:
        return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    grayscale = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    return np.expand_dims(grayscale, axis=-1)


# ─── Record constructors ──────────────────────────────────────────────────────

def _split_sorted(sample_ids: List[str], train_count: int, val_count: int) -> Dict[str, List[str]]:
    test_count = max(0, len(sample_ids) - train_count - val_count)
    return {
        "train": sample_ids[:train_count],
        "val": sample_ids[train_count: train_count + val_count],
        "test": sample_ids[-test_count:] if test_count > 0 else [],
    }


def _new_record(
    sample_id: str,
    image_path: str,
    landmarks: np.ndarray,
    scaling_factor: np.ndarray,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "landmarks": landmarks.astype(np.float64),
        "physical_scaling_factor": scaling_factor.astype(np.float64),
    }


# ─── Dataset parsers ──────────────────────────────────────────────────────────

def _parse_chest_records(dataset_root: str) -> Dict[str, List[Dict[str, Any]]]:
    chest_root = os.path.join(dataset_root, "chest")
    image_dir = os.path.join(chest_root, "pngs")
    label_dir = os.path.join(chest_root, "labels")

    stems = [f[:-4] for f in sorted(os.listdir(image_dir)) if f.endswith(".png")]
    split_map = _split_sorted(stems, train_count=195, val_count=34)

    records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name, sample_ids in split_map.items():
        for sample_id in sample_ids:
            image_path = os.path.join(image_dir, f"{sample_id}.png")
            image_h, image_w = read_image_shape(image_path)

            points: List[Tuple[float, float]] = []
            with open(os.path.join(label_dir, f"{sample_id}.txt"), "r") as handle:
                point_count = int(handle.readline().strip())
                for _ in range(point_count):
                    rx, ry = [float(t) for t in handle.readline().split()]
                    points.append((rx * image_w, ry * image_h))

            records[split_name].append(
                _new_record(
                    sample_id=sample_id,
                    image_path=image_path,
                    landmarks=np.asarray(points, dtype=np.float64),
                    scaling_factor=np.asarray([1.0, 1.0], dtype=np.float64),
                )
            )
    return records


def _parse_hand_records(dataset_root: str, image_size: Tuple[int, int]) -> Dict[str, List[Dict[str, Any]]]:
    hand_root = os.path.join(dataset_root, "hand")
    image_dir = os.path.join(hand_root, "jpg")
    labels_df = pd.read_csv(os.path.join(hand_root, "labels", "all.csv"), header=None, index_col=0)

    stems = [f[:-4] for f in sorted(os.listdir(image_dir)) if f.endswith(".jpg")]
    valid_indices = set(labels_df.index.tolist())
    stems = [s for s in stems if int(s) in valid_indices]

    split_map = {"train": stems[:550], "val": stems[550:609], "test": stems[609:]}
    wrist_width_mm = 50.0
    records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split_name, sample_ids in split_map.items():
        for sample_id in sample_ids:
            image_path = os.path.join(image_dir, f"{sample_id}.jpg")
            image_h, image_w = read_image_shape(image_path)
            row = [float(v) for v in list(labels_df.loc[int(sample_id), :])]
            landmarks = np.asarray([(row[i], row[i + 1]) for i in range(0, len(row), 2)], dtype=np.float64)

            wrist_width_pixels = compute_euclidean_distance(landmarks[0], landmarks[4])
            mm_per_pixel = wrist_width_mm / max(wrist_width_pixels, 1e-8)
            scale_x, scale_y = get_scale_factor((image_h, image_w), image_size)
            scaling = np.asarray([scale_x * mm_per_pixel, scale_y * mm_per_pixel], dtype=np.float64)

            records[split_name].append(
                _new_record(sample_id=sample_id, image_path=image_path, landmarks=landmarks, scaling_factor=scaling)
            )
    return records


def _parse_cephalo_records(dataset_root: str, image_size: Tuple[int, int]) -> Dict[str, List[Dict[str, Any]]]:
    ceph_root = os.path.join(dataset_root, "cephalo")
    image_dir = os.path.join(ceph_root, "jpg")
    junior_dir = os.path.join(ceph_root, "400_junior")
    senior_dir = os.path.join(ceph_root, "400_senior")

    sample_ids = [f[:-4] for f in sorted(os.listdir(image_dir)) if f.endswith(".jpg")]
    split_map = {"train": sample_ids[:130], "val": sample_ids[130:150], "test": sample_ids[150:400]}

    records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split_name, split_ids in split_map.items():
        for sample_id in split_ids:
            image_path = os.path.join(image_dir, f"{sample_id}.jpg")
            image_h, image_w = read_image_shape(image_path)

            landmarks: List[Tuple[float, float]] = []
            with open(os.path.join(junior_dir, f"{sample_id}.txt"), "r") as jf, \
                 open(os.path.join(senior_dir, f"{sample_id}.txt"), "r") as sf:
                for _ in range(19):
                    jl, sl = jf.readline().strip(), sf.readline().strip()
                    if not jl or not sl:
                        break
                    jx, jy = [float(t) for t in jl.split(",")]
                    sx, sy = [float(t) for t in sl.split(",")]
                    landmarks.append(((jx + sx) * 0.5, (jy + sy) * 0.5))

            lm_np = np.asarray(landmarks, dtype=np.float64)
            scale_x, scale_y = get_scale_factor((image_h, image_w), image_size)
            scaling = np.asarray([scale_x * 0.1, scale_y * 0.1], dtype=np.float64)

            records[split_name].append(
                _new_record(sample_id=sample_id, image_path=image_path, landmarks=lm_np, scaling_factor=scaling)
            )
    return records




# ─── Split loader ─────────────────────────────────────────────────────────────

def load_dataset_splits(cfg: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """Parse the requested dataset and return (splits_dict, num_landmarks)."""
    name = str(cfg["dataset"]).lower().strip()
    if name == "chest":
        splits = _parse_chest_records(cfg["dataset_root"])
    elif name == "hand":
        splits = _parse_hand_records(cfg["dataset_root"], cfg["image_size"])
    elif name == "cephalo":
        splits = _parse_cephalo_records(cfg["dataset_root"], cfg["image_size"])
    else:
        raise ValueError(f"Unsupported dataset '{cfg['dataset']}'")

    num_landmarks = DATASET_NUM_LANDMARKS[name]
    if splits["train"]:
        num_landmarks = int(len(splits["train"][0]["landmarks"]))
    return splits, num_landmarks


def sorted_records_by_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: r["sample_id"])


def subset_records(records_by_id: Dict[str, Dict[str, Any]], sample_ids: Sequence[str]) -> List[Dict[str, Any]]:
    return [records_by_id[sid] for sid in sample_ids]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class LandmarkDataset(Dataset):
    """Torch dataset supporting oracle labels and optional pseudo-labels.

    Pseudo-labels are used when a sample_id appears in ``pseudo_labels``.
    Each pseudo entry is ``{"landmarks": [[x, y], ...]}``. Oracle samples use
    keypoint-aware transforms with fallback (advanced -> optimal -> simple ->
    base). Pseudo samples use image-only preprocessing because pseudo landmarks
    are already stored in post-preprocessing coordinates.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str,
        image_size: Tuple[int, int],
        sigma: float,
        training_mode: bool,
        channels: int = 1,
        pseudo_labels: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.records = records
        self.dataset_name = str(dataset_name).lower().strip()
        self.image_size = tuple(image_size)
        self.sigma = float(sigma)
        self.training_mode = bool(training_mode)
        self.channels = int(channels)
        self.pseudo_labels = pseudo_labels or {}

        self._resize_pad_transforms = [
            A.LongestMaxSize(max_size=max(self.image_size)),
            A.PadIfNeeded(min_height=self.image_size[0], min_width=self.image_size[1], border_mode=cv2.BORDER_CONSTANT),
        ]
        self._normalize_transforms = [
            A.Normalize(normalization="min_max"),
            ToTensorV2(),
        ]
        self._preprocessing = self._resize_pad_transforms + self._normalize_transforms
        self.keypoint_base_transform = A.Compose(
            self._preprocessing,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        self._resize_pad_image_only = A.Compose(self._resize_pad_transforms)

    def _intensity_augmentations(self) -> List[A.BasicTransform]:
        return [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.6),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.04, 0.15), per_channel=True),
                    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.5)),
                ],
                p=0.4,
            ),
        ]

    def _simple_augmentations(self) -> List[A.BasicTransform]:
        return [
            A.Affine(
                scale=(0.94, 1.10),
                translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                rotate=(-8, 8),
                shear=(-2, 2),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            A.RandomGamma(gamma_limit=(92, 108), p=0.4),
            A.CLAHE(clip_limit=(1, 2), tile_grid_size=(8, 8), p=0.15),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.015, 0.07), per_channel=True),
                    A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.5, 1.0)),
                ],
                p=0.2,
            ),
        ]

    def _optimal_augmentations(self) -> List[A.BasicTransform]:
        return [
            A.Affine(
                scale=(0.85, 1.20),
                translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                rotate=(-15, 15),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8,
            ),
            A.ElasticTransform(alpha=42, sigma=8, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.30),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.11,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.20,
            ),
        ]

    def _advanced_augmentations(self) -> List[A.BasicTransform]:
        return [
            A.Affine(
                scale=(0.80, 1.25),
                translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)},
                rotate=(-18, 18),
                shear=(-6, 6),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8,
            ),
            A.ElasticTransform(alpha=48, sigma=9, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.35),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.14,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.25,
            ),
        ] + self._intensity_augmentations()

    def _augmentations_for_level(self, level: str) -> List[A.BasicTransform]:
        if level == "base":
            return []
        if level == "simple":
            return self._simple_augmentations()
        if level == "optimal":
            return self._optimal_augmentations()
        if level == "advanced":
            return self._advanced_augmentations()
        raise ValueError(f"Unknown augmentation level '{level}'")

    def _keypoint_transform_for_level(self, level: str, skip_resize_pad: bool = False) -> A.Compose:
        augmentations = self._augmentations_for_level(level)
        preprocessing = self._normalize_transforms if skip_resize_pad else self._preprocessing
        return A.Compose(
            augmentations + preprocessing,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def _apply_with_fallback(
        self,
        image_np: np.ndarray,
        landmarks: np.ndarray,
        skip_resize_pad: bool = False,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        if self.training_mode:
            levels = ["advanced", "optimal", "simple", "base"]
        else:
            levels = ["base"]

        for level in levels:
            transform = self._keypoint_transform_for_level(level, skip_resize_pad=skip_resize_pad)
            transformed = transform(image=image_np, keypoints=landmarks.tolist())
            transformed_landmarks = np.asarray(transformed["keypoints"], dtype=np.float64)
            if transformed_landmarks.shape[0] == landmarks.shape[0]:
                return transformed["image"].float(), transformed_landmarks

        raise RuntimeError(
            f"Failed to preserve landmarks after fallback transforms for dataset={self.dataset_name}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def _oracle_item(self, record: Dict[str, Any], image_np: np.ndarray):
        img_t, lms = self._apply_with_fallback(image_np=image_np, landmarks=record["landmarks"])
        K = lms.shape[0]
        return img_t, lms, 1.0, np.ones(K, dtype=np.float32)

    def _pseudo_item(self, pseudo_info: Dict[str, Any], image_np: np.ndarray):
        image_resized = self._resize_pad_image_only(image=image_np)["image"]
        landmarks = np.asarray(pseudo_info["landmarks"], dtype=np.float64)
        img_t, lms = self._apply_with_fallback(
            image_np=image_resized,
            landmarks=landmarks,
            skip_resize_pad=True,
        )
        K = lms.shape[0]
        sample_weight = float(pseudo_info.get("weight", 1.0))
        mask_list = pseudo_info.get("landmark_mask")
        if mask_list is not None:
            landmark_mask = np.asarray(list(mask_list), dtype=np.float32)
        else:
            landmark_mask = np.ones(K, dtype=np.float32)
        return img_t, lms, sample_weight, landmark_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample_id = record["sample_id"]
        image_np = read_image(record["image_path"], channels=self.channels)

        pseudo_info = self.pseudo_labels.get(sample_id)
        if pseudo_info is None:
            image_tensor, landmarks, sample_weight, landmark_mask = self._oracle_item(record, image_np)
        else:
            image_tensor, landmarks, sample_weight, landmark_mask = self._pseudo_item(pseudo_info, image_np)

        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, self.image_size[1] - 1)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, self.image_size[0] - 1)
        heatmaps = keypoints2heatmaps(landmarks=landmarks, img_size=self.image_size, sigma=self.sigma)

        return {
            "name": sample_id,
            "image": image_tensor,
            "landmarks": torch.from_numpy(landmarks).double(),
            "target": torch.from_numpy(heatmaps).float(),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "landmark_mask": torch.from_numpy(landmark_mask).float(),
            "physical_scaling_factor": torch.from_numpy(record["physical_scaling_factor"]).double(),
        }


# ─── DataLoader factory ───────────────────────────────────────────────────────

def create_dataloader(
    records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    shuffle: bool,
    seed: int,
    pseudo_labels: Optional[Dict[str, Dict[str, Any]]] = None,
) -> DataLoader:
    dataset = LandmarkDataset(
        records=records,
        dataset_name=cfg["dataset"],
        image_size=cfg["image_size"],
        sigma=cfg["sigma"],
        training_mode=bool(shuffle),
        channels=cfg["image_channels"],
        pseudo_labels=pseudo_labels,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
    )
