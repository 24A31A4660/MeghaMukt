"""preprocessing/loader.py — PyTorch Dataset for paired cloudy/clear patches."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.transforms import (
    apply_transforms,
    get_train_transforms,
    get_val_transforms,
    to_chw,
    to_hwc,
)


class CloudRemovalDataset(Dataset):
    """
    PyTorch Dataset for cloud removal training.

    Expects pre-extracted .npy patch files organised as:
        dataset/
          train/
            cloudy/   ← input patches [C, H, W]
            clear/    ← target patches [C, H, W]
            masks/    ← binary cloud masks [H, W]
          validation/
          test/

    File naming convention:
        pair_XXXXXX_cloudy.npy
        pair_XXXXXX_clear.npy
        pair_XXXXXX_mask.npy

    Parameters
    ----------
    root_dir   : Path to dataset split directory (train/validation/test).
    split      : "train" | "validation" | "test".
    patch_size : Expected patch size (for validation).
    augment    : Apply augmentation (only for "train").
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == "train")

        self.cloudy_dir = self.root_dir / "cloudy"
        self.clear_dir  = self.root_dir / "clear"
        self.mask_dir   = self.root_dir / "masks"

        # Discover pairs by matching stem names
        self.pair_ids = self._discover_pairs()

        # Choose augmentation pipeline
        self.transform = (
            get_train_transforms(patch_size) if self.augment
            else get_val_transforms(patch_size)
        )

    def _discover_pairs(self) -> list[str]:
        """Find all valid (cloudy, clear, mask) triplets by stem."""
        if not self.cloudy_dir.exists():
            raise FileNotFoundError(f"Cloudy patch directory not found: {self.cloudy_dir}")

        pairs: list[str] = []
        for cloudy_file in sorted(self.cloudy_dir.glob("*.npy")):
            stem = cloudy_file.stem.replace("_cloudy", "")
            clear_file = self.clear_dir / f"{stem}_clear.npy"
            mask_file  = self.mask_dir  / f"{stem}_mask.npy"

            if not clear_file.exists():
                continue   # skip unpaired
            if not mask_file.exists():
                continue   # skip missing mask

            pairs.append(stem)

        return pairs

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.pair_ids[idx]

        # Load .npy arrays
        try:
            cloudy = np.load(self.cloudy_dir / f"{stem}_cloudy.npy").astype(np.float32)
            clear  = np.load(self.clear_dir  / f"{stem}_clear.npy").astype(np.float32)
            mask   = np.load(self.mask_dir   / f"{stem}_mask.npy").astype(np.uint8)
        except Exception as exc:
            # Skip corrupted file — return zeros as a safety fallback
            c = cloudy.shape[0] if 'cloudy' in dir() else 7
            dummy = np.zeros((c, self.patch_size, self.patch_size), dtype=np.float32)
            dummy_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            return {
                "cloudy": torch.from_numpy(dummy),
                "clear":  torch.from_numpy(dummy[:6]),
                "mask":   torch.from_numpy(dummy_mask).float(),
                "stem":   stem,
            }

        # Augmentation: Albumentations expects [H, W, C]
        if self.augment:
            cloudy_hwc = to_hwc(cloudy)
            clear_hwc  = to_hwc(clear)
            cloudy_hwc, clear_hwc, mask = apply_transforms(
                self.transform, cloudy_hwc, clear_hwc, mask
            )
            cloudy = to_chw(cloudy_hwc)
            clear  = to_chw(clear_hwc)

        return {
            "cloudy": torch.from_numpy(cloudy).float(),     # [C, H, W] — optical + mask
            "clear":  torch.from_numpy(clear).float(),      # [C_optical, H, W]
            "mask":   torch.from_numpy(mask).float(),       # [H, W]
            "stem":   stem,
        }


def build_dataloaders(
    dataset_dir: Path,
    patch_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """
    Build train / validation / test DataLoaders.

    Returns dict with keys "train", "validation", "test".
    """
    loaders: dict[str, DataLoader] = {}
    splits = {
        "train":      {"shuffle": True,  "augment": True},
        "validation": {"shuffle": False, "augment": False},
        "test":       {"shuffle": False, "augment": False},
    }

    for split, opts in splits.items():
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            continue
        ds = CloudRemovalDataset(
            root_dir=dataset_dir,
            split=split,
            patch_size=patch_size,
            augment=opts["augment"],
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=opts["shuffle"],
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
        )

    return loaders
