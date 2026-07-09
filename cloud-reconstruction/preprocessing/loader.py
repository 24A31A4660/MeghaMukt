"""preprocessing/loader.py — PyTorch Dataset for paired cloudy/clear patches."""
from __future__ import annotations

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


def _ensure_patch_size(
    cloudy: np.ndarray,   # [C, H, W]
    clear:  np.ndarray,   # [C, H, W]
    mask:   np.ndarray,   # [H, W]
    target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center-crop or zero-pad all arrays to exactly (target × target).

    The dataset may contain patches of mixed sizes (e.g. 256 and 384) if
    it was built from multiple extraction runs.  DataLoader.stack() requires
    all items in a batch to be the same shape, so we normalise here.

    Args:
        cloudy, clear : [C, H, W] float32 in [0, 1]
        mask          : [H, W] uint8 binary
        target        : desired spatial size

    Returns normalised (cloudy, clear, mask) all at (target × target).
    """
    h, w = cloudy.shape[1], cloudy.shape[2]
    if h == target and w == target:
        return cloudy, clear, mask

    def _crop_or_pad(arr: np.ndarray, is_mask: bool) -> np.ndarray:
        # arr is [C, H, W] or [H, W]
        spatial = arr if is_mask else arr
        ah = spatial.shape[-2]
        aw = spatial.shape[-1]

        # Center-crop if too large
        if ah > target:
            y0 = (ah - target) // 2
            spatial = spatial[..., y0:y0 + target, :]
        if aw > target:
            x0 = (aw - target) // 2
            spatial = spatial[..., :, x0:x0 + target]

        # Zero-pad if too small
        pad_h = max(0, target - spatial.shape[-2])
        pad_w = max(0, target - spatial.shape[-1])
        if pad_h > 0 or pad_w > 0:
            ph0, ph1 = pad_h // 2, pad_h - pad_h // 2
            pw0, pw1 = pad_w // 2, pad_w - pad_w // 2
            if is_mask:
                spatial = np.pad(spatial, ((ph0, ph1), (pw0, pw1)), mode="constant")
            else:
                spatial = np.pad(spatial, ((0, 0), (ph0, ph1), (pw0, pw1)), mode="constant")
        return spatial

    cloudy = _crop_or_pad(cloudy, is_mask=False)
    clear  = _crop_or_pad(clear,  is_mask=False)
    mask   = _crop_or_pad(mask,   is_mask=True)
    return cloudy, clear, mask



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
        patch_size: int = 384,
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
        except Exception:
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

        # ── Enforce uniform spatial size ──────────────────────────────────
        # The dataset may contain patches of varying sizes (e.g. 256 and 384).
        # DataLoader.stack() requires all items in a batch to be the same size.
        # Strategy: center-crop if larger, zero-pad if smaller.
        cloudy, clear, mask = _ensure_patch_size(cloudy, clear, mask, self.patch_size)

        return {
            "cloudy": torch.from_numpy(cloudy).float(),     # [C, H, W] — optical + mask
            "clear":  torch.from_numpy(clear).float(),      # [C_optical, H, W]
            "mask":   torch.from_numpy(mask).float(),       # [H, W]
            "stem":   stem,
        }


def build_dataloaders(
    dataset_dir,
    patch_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    """
    Build train / validation / test DataLoaders.

    Accepts either:
      - A Path/str to the dataset directory, or
      - A full config dict (extracts dataset_dir and training params from it).

    Returns a dict-like object with keys "train", "validation", "test" that also
    supports tuple unpacking: ``train_loader, val_loader, test_loader = build_dataloaders(...)``
    """
    # ── Extract config if a dict was passed ──
    if isinstance(dataset_dir, dict):
        cfg = dataset_dir
        if "dataset" in cfg and "output_dir" in cfg["dataset"]:
            root_path = Path(cfg["dataset"]["output_dir"])
        elif "root" in cfg:
            root_path = Path(cfg["root"])
        else:
            root_path = Path(str(cfg))

        if "patch" in cfg and "size" in cfg["patch"]:
            patch_size = cfg["patch"]["size"]
        if "training" in cfg:
            batch_size = cfg["training"].get("batch_size", batch_size)
            num_workers = cfg["training"].get("num_workers", num_workers)
            pin_memory = cfg["training"].get("pin_memory", pin_memory)
            prefetch_factor = cfg["training"].get("prefetch_factor", prefetch_factor)

        dataset_dir = root_path

    dataset_dir = Path(dataset_dir)

    # ── Validate folders exist ──
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder missing: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder missing: {val_dir}")

    loaders: dict[str, DataLoader] = {}
    splits = {
        "train":      {"shuffle": True,  "augment": True},
        "validation": {"shuffle": False, "augment": False},
        "test":       {"shuffle": False, "augment": False},
    }

    image_counts = {}

    for split, opts in splits.items():
        split_dir = dataset_dir / split
        if not split_dir.exists():
            image_counts[split] = 0
            continue

        ds = CloudRemovalDataset(
            root_dir=dataset_dir,
            split=split,
            patch_size=patch_size,
            augment=opts["augment"],
        )
        image_counts[split] = len(ds)
        kwargs = {
            "batch_size": batch_size,
            "shuffle": opts["shuffle"],
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": (split == "train"),
            "persistent_workers": (num_workers > 0),
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor

        loaders[split] = DataLoader(ds, **kwargs)

    # ── Print stats ──
    print(f"Dataset Root: {dataset_dir}")
    print(f"Number of train images: {image_counts.get('train', 0)}")
    print(f"Number of validation images: {image_counts.get('validation', 0)}")
    print(f"Number of test images: {image_counts.get('test', 0)}")

    # Return a dict that also supports tuple-unpacking for backward compatibility
    class _UnpackableDict(dict):
        """Dict subclass that yields (train, validation, test) loaders when unpacked."""
        def __iter__(self):
            yield self.get("train")
            yield self.get("validation")
            yield self.get("test")

    return _UnpackableDict(loaders)
