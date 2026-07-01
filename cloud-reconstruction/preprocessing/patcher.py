"""preprocessing/patcher.py — Patch extraction and Gaussian-weighted patch merging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Patch:
    data: np.ndarray      # [C, patch_h, patch_w]
    row: int              # top-left row in full image
    col: int              # top-left col in full image
    patch_h: int
    patch_w: int


class PatchExtractor:
    """
    Extract overlapping patches from a [C, H, W] image tensor.

    Parameters
    ----------
    patch_size : Patch height and width (square).
    stride     : Step between patches. Use stride < patch_size for overlap.
    """

    def __init__(self, patch_size: int = 256, stride: int = 128) -> None:
        self.patch_size = patch_size
        self.stride = stride

    def extract(self, image: np.ndarray) -> list[Patch]:
        """
        Extract all patches from a [C, H, W] array.

        Returns list of Patch objects.
        """
        c, h, w = image.shape
        patches: list[Patch] = []

        for r in range(0, h - self.patch_size + 1, self.stride):
            for c_col in range(0, w - self.patch_size + 1, self.stride):
                patch_data = image[
                    :,
                    r : r + self.patch_size,
                    c_col : c_col + self.patch_size,
                ]
                patches.append(Patch(
                    data=patch_data.copy(),
                    row=r,
                    col=c_col,
                    patch_h=self.patch_size,
                    patch_w=self.patch_size,
                ))

        return patches

    def extract_with_positions(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Returns (stacked_patches [N, C, H, W], positions [(row, col), ...]).
        Useful for batch inference.
        """
        patches = self.extract(image)
        stacked = np.stack([p.data for p in patches], axis=0)
        positions = [(p.row, p.col) for p in patches]
        return stacked, positions

    def iter_patches(self, image: np.ndarray) -> Iterator[Patch]:
        """Iterate over patches one at a time (memory-efficient)."""
        _, h, w = image.shape
        for r in range(0, h - self.patch_size + 1, self.stride):
            for c_col in range(0, w - self.patch_size + 1, self.stride):
                patch_data = image[
                    :,
                    r : r + self.patch_size,
                    c_col : c_col + self.patch_size,
                ]
                yield Patch(patch_data.copy(), r, c_col, self.patch_size, self.patch_size)


class PatchMerger:
    """
    Reconstruct a full image from overlapping patches using Gaussian-weighted
    blending to eliminate seam artefacts at patch boundaries.

    Parameters
    ----------
    full_shape : (C, H, W) of the target output image.
    patch_size : Patch size used during extraction.
    """

    def __init__(
        self,
        full_shape: tuple[int, int, int],
        patch_size: int = 256,
    ) -> None:
        self.full_shape = full_shape  # (C, H, W)
        self.patch_size = patch_size
        self._weight_map = self._gaussian_weight(patch_size)

    @staticmethod
    def _gaussian_weight(size: int) -> np.ndarray:
        """
        Create a 2D Gaussian weight map for a patch.
        Higher weights at the centre, tapering to near-zero at edges.
        Shape: [size, size], float32.
        """
        sigma = size / 4.0
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
        gauss_2d = np.outer(gauss_1d, gauss_1d).astype(np.float32)
        gauss_2d /= gauss_2d.max()  # normalise to [0, 1]
        return gauss_2d

    def merge(
        self,
        patches: list[np.ndarray],
        positions: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Merge reconstructed patches back into a full image.
        Processes one band at a time to minimize memory usage.

        Parameters
        ----------
        patches   : List of [C, H, W] reconstructed patch arrays.
        positions : List of (row, col) top-left positions.

        Returns
        -------
        merged : [C, H, W] float32 full-image reconstruction.
        """
        c, full_h, full_w = self.full_shape
        w = self._weight_map  # [patch_h, patch_w]

        # Build weight accumulator once (shared across bands)
        weight_acc = np.zeros((full_h, full_w), dtype=np.float32)
        for patch, (r, col) in zip(patches, positions):
            ph, pw = patch.shape[1], patch.shape[2]
            weight_acc[r:r+ph, col:col+pw] += w[:ph, :pw]
        weight_acc = np.maximum(weight_acc, 1e-8)

        # Process one band at a time to avoid allocating full [C, H, W]
        merged = np.zeros((c, full_h, full_w), dtype=np.float32)

        for band_idx in range(c):
            band_acc = np.zeros((full_h, full_w), dtype=np.float32)
            for patch, (r, col) in zip(patches, positions):
                ph, pw = patch.shape[1], patch.shape[2]
                band_acc[r:r+ph, col:col+pw] += patch[band_idx, :ph, :pw] * w[:ph, :pw]
            merged[band_idx] = band_acc / weight_acc
            del band_acc

        return merged

    def merge_with_cloud_blend(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        cloud_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Blend reconstructed image with original using cloud mask.

        CRITICAL: Cloud-free pixels in the original are NEVER modified.
        Only cloud-masked pixels are replaced with reconstruction.

        Parameters
        ----------
        original      : [C, H, W] original cloudy image.
        reconstructed : [C, H, W] model output.
        cloud_mask    : [H, W] binary mask (1=cloud, 0=clear).

        Returns
        -------
        blended : [C, H, W] — original where clear, reconstructed where cloudy.
        """
        mask = cloud_mask[np.newaxis].astype(np.float32)  # [1, H, W]
        blended = original * (1 - mask) + reconstructed * mask
        return blended.astype(np.float32)


def filter_patches_by_cloud(
    patches: list[Patch],
    cloud_mask: np.ndarray,
    min_cloud: float = 0.05,
    max_cloud: float = 0.99,
) -> list[Patch]:
    """
    Filter patches to only keep those with meaningful cloud coverage.
    Skip patches that are nearly all cloud (no useful ground truth)
    or nearly all clear (reconstruction not needed for training).
    """
    kept: list[Patch] = []
    for p in patches:
        mask_crop = cloud_mask[p.row:p.row+p.patch_h, p.col:p.col+p.patch_w]
        frac = float(np.mean(mask_crop))
        if min_cloud <= frac <= max_cloud:
            kept.append(p)
    return kept
