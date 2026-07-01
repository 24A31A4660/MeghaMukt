"""preprocessing/transforms.py — Albumentations augmentation pipeline for training."""
from __future__ import annotations

from typing import Optional

import albumentations as A
import numpy as np


def get_train_transforms(patch_size: int = 256) -> A.Compose:
    """
    Training augmentation pipeline.
    Uses only spatial transforms (flips, rotations) which are safe for
    multi-spectral images with arbitrary channel counts (6 and 7 channels).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={
            "clear_image": "image",
            "cloud_mask":  "mask",
        },
        is_check_shapes=False,
    )


def get_val_transforms(patch_size: int = 256) -> A.Compose:
    """Validation: no augmentation, just ensure correct format."""
    return A.Compose(
        [],
        additional_targets={
            "clear_image": "image",
            "cloud_mask":  "mask",
        },
        is_check_shapes=False,
    )


def apply_transforms(
    transform: A.Compose,
    cloudy: np.ndarray,
    clear: np.ndarray,
    cloud_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply an Albumentations transform to a (cloudy, clear, mask) triplet.

    Parameters
    ----------
    transform   : Albumentations Compose pipeline.
    cloudy      : [H, W, C] float32 — cloudy input image.
    clear       : [H, W, C] float32 — cloud-free target image.
    cloud_mask  : [H, W] uint8 — binary cloud mask.

    Returns
    -------
    Augmented (cloudy, clear, cloud_mask) in the same format.
    """
    result = transform(
        image=cloudy,
        clear_image=clear,
        cloud_mask=cloud_mask,
    )
    return (
        result["image"],
        result["clear_image"],
        result["cloud_mask"],
    )


def to_chw(image: np.ndarray) -> np.ndarray:
    """Convert [H, W, C] → [C, H, W] for PyTorch."""
    return np.transpose(image, (2, 0, 1)).copy()


def to_hwc(image: np.ndarray) -> np.ndarray:
    """Convert [C, H, W] → [H, W, C] for Albumentations."""
    return np.transpose(image, (1, 2, 0)).copy()
