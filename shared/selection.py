"""Generic selection utilities shared across methods."""
from typing import Sequence, List
import numpy as np


def sorted_ids(items: Sequence[str]) -> List[str]:
    return sorted(list(items))


def select_initial_labeled(train_ids: Sequence[str], n_init: int, seed: int) -> List[str]:
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")
    if n_init > len(train_ids):
        raise ValueError(
            f"Requested n_init={n_init} but train split has only {len(train_ids)} samples"
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.asarray(list(train_ids)), size=n_init, replace=False)
    return sorted(selected.tolist())
