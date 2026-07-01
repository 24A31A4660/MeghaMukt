"""preprocessing/rice_loader.py — PyTorch Dataset for RICE (RGB) dataset."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from preprocessing.transforms import (
    apply_transforms,
    get_train_transforms,
    get_val_transforms,
    to_chw,
    to_hwc,
)

class RICEDataset(Dataset):
    """
    PyTorch Dataset for RICE 1 and 2 cloud removal training.

    Expects dataset structured as:
        dataset/
          RICE1/
            cloud/
            label/
          RICE2/
            cloud/
            label/
            mask/   (optional)
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
        append_mask: bool = False,
        val_split_ratio: float = 0.1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == "train")
        self.append_mask = append_mask

        self.samples = self._discover_samples()

        # Split into train/val
        # Deterministic split based on sorted names
        split_idx = int(len(self.samples) * (1.0 - val_split_ratio))
        if self.split == "train":
            self.samples = self.samples[:split_idx]
        elif self.split == "validation":
            self.samples = self.samples[split_idx:]

        self.transform = (
            get_train_transforms(patch_size) if self.augment
            else get_val_transforms(patch_size)
        )

    def _discover_samples(self) -> list[dict[str, Path]]:
        """Find all (cloud, label, mask) tuples."""
        samples = []
        for rice_sub in ["RICE1", "RICE2"]:
            sub_dir = self.root_dir / rice_sub
            if not sub_dir.exists():
                continue
                
            cloud_dir = sub_dir / "cloud"
            label_dir = sub_dir / "label"
            mask_dir = sub_dir / "mask"

            if not cloud_dir.exists() or not label_dir.exists():
                continue

            for cloud_file in sorted(cloud_dir.glob("*.*")):
                if cloud_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    continue

                stem = cloud_file.stem
                # Check for corresponding label
                label_file = None
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    candidate = label_dir / f"{stem}{ext}"
                    if candidate.exists():
                        label_file = candidate
                        break

                if not label_file:
                    continue

                # Check for mask
                mask_file = None
                if mask_dir.exists():
                    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                        candidate = mask_dir / f"{stem}{ext}"
                        if candidate.exists():
                            mask_file = candidate
                            break

                samples.append({
                    "cloud": cloud_file,
                    "label": label_file,
                    "mask": mask_file,
                    "stem": f"{rice_sub}_{stem}"
                })
                
        return sorted(samples, key=lambda x: x["stem"])

    def _generate_mask(self, cloudy_arr: np.ndarray, clear_arr: np.ndarray) -> np.ndarray:
        """Generate a basic mask by thresholding brightness difference if no mask exists."""
        # Simple absolute difference in grayscale
        diff = np.abs(cloudy_arr.mean(axis=2) - clear_arr.mean(axis=2))
        mask = (diff > 0.1).astype(np.uint8)
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load RGB images [H, W, C] normalized to [0, 1]
        try:
            cloudy_img = Image.open(sample["cloud"]).convert("RGB")
            clear_img = Image.open(sample["label"]).convert("RGB")
            
            # Resize if needed to ensure consistency before albumentations
            cloudy_img = cloudy_img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            clear_img = clear_img.resize((self.patch_size, self.patch_size), Image.BILINEAR)

            cloudy = np.array(cloudy_img, dtype=np.float32) / 255.0
            clear = np.array(clear_img, dtype=np.float32) / 255.0

            if sample["mask"] is not None:
                mask_img = Image.open(sample["mask"]).convert("L")
                mask_img = mask_img.resize((self.patch_size, self.patch_size), Image.NEAREST)
                mask = np.array(mask_img, dtype=np.uint8)
                mask = (mask > 127).astype(np.uint8)
            else:
                mask = self._generate_mask(cloudy, clear)

        except Exception as exc:
            # Fallback for corrupted file
            dummy = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
            dummy_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            cloudy = clear = dummy
            mask = dummy_mask

        # Augmentation: Albumentations expects [H, W, C]
        if self.augment:
            cloudy, clear, mask = apply_transforms(self.transform, cloudy, clear, mask)

        # Convert to [C, H, W]
        cloudy_chw = to_chw(cloudy)
        clear_chw = to_chw(clear)

        if self.append_mask:
            # Append mask as the 4th channel
            mask_chw = np.expand_dims(mask, axis=0).astype(np.float32)
            cloudy_chw = np.concatenate([cloudy_chw, mask_chw], axis=0)

        return {
            "cloudy": torch.from_numpy(cloudy_chw).float(),  # [3 or 4, H, W]
            "clear":  torch.from_numpy(clear_chw).float(),   # [3, H, W]
            "mask":   torch.from_numpy(mask).float(),        # [H, W]
            "stem":   sample["stem"],
        }


def build_rice_dataloaders(
    dataset_dir: Path,
    patch_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    append_mask: bool = False,
) -> dict[str, DataLoader]:
    """
    Build train / validation DataLoaders for RICE dataset.
    """
    loaders: dict[str, DataLoader] = {}
    splits = {
        "train":      {"shuffle": True,  "augment": True},
        "validation": {"shuffle": False, "augment": False},
    }

    for split, opts in splits.items():
        ds = RICEDataset(
            root_dir=dataset_dir,
            split=split,
            patch_size=patch_size,
            augment=opts["augment"],
            append_mask=append_mask,
        )
        if len(ds) == 0:
            continue
            
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=opts["shuffle"],
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

    return loaders
