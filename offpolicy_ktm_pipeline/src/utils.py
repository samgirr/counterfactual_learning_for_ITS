from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set numpy/python/torch seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device(device: str | None = None) -> torch.device:
    """
    Resolve target torch device from user value or hardware availability.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure parent directory exists for a file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

