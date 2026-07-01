"""preprocessing/cloud_detector.py — SCL-based and threshold-based cloud mask generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Sentinel-2 SCL class values that correspond to clouds/cloud shadows
SCL_CLOUD_CLASSES = {
    3,   # Cloud shadow
    8,   # Cloud medium probability
    9,   # Cloud high probability
    10,  # Thin cirrus
}
SCL_CLOUD_SHADOW = {3}
SCL_CLOUD_ONLY = {8, 9, 10}


@dataclass
class CloudMaskResult:
    mask: np.ndarray          # Binary [H, W] uint8  0=clear 1=cloud
    cloud_fraction: float     # Fraction of pixels that are cloudy
    method: str               # "scl" | "threshold"
    confidence: float         # 0–1, higher = more certain


class SCLCloudDetector:
    """
    Generates a binary cloud mask from the Sentinel-2 SCL (Scene Classification Layer).

    SCL band encoding (ESA L2A):
      0  = No data
      1  = Saturated / defective
      2  = Dark area pixels
      3  = Cloud shadows           ← masked
      4  = Vegetation
      5  = Bare soil
      6  = Water
      7  = Unclassified
      8  = Cloud medium prob       ← masked
      9  = Cloud high prob         ← masked
      10 = Thin cirrus             ← masked
      11 = Snow / ice
    """

    def __init__(self, include_shadow: bool = True, dilate_pixels: int = 3) -> None:
        """
        Parameters
        ----------
        include_shadow : Also mask cloud shadow pixels (SCL=3).
        dilate_pixels  : Morphological dilation radius to buffer cloud edges (0 = no dilation).
        """
        self.include_shadow = include_shadow
        self.dilate_pixels = dilate_pixels
        self.cloud_classes = SCL_CLOUD_ONLY | (SCL_CLOUD_SHADOW if include_shadow else set())

    def generate(self, scl: np.ndarray) -> CloudMaskResult:
        """
        Parameters
        ----------
        scl : SCL band array [H, W] with integer class values.

        Returns
        -------
        CloudMaskResult
        """
        mask = np.zeros(scl.shape, dtype=np.uint8)
        for cls in self.cloud_classes:
            mask[scl == cls] = 1

        # Morphological dilation to buffer cloud edges
        if self.dilate_pixels > 0:
            mask = self._dilate(mask, self.dilate_pixels)

        cloud_fraction = float(np.mean(mask))
        return CloudMaskResult(
            mask=mask,
            cloud_fraction=cloud_fraction,
            method="scl",
            confidence=0.95,  # SCL is official ESA product — high confidence
        )

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        """Simple binary dilation using scipy."""
        try:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
            return binary_dilation(mask, structure=struct).astype(np.uint8)
        except ImportError:
            return mask  # fallback: no dilation


class ThresholdCloudDetector:
    """
    Fallback cloud detector using brightness + whiteness thresholds.
    Used during inference when SCL is not available.

    Works on RGB (or any 3-band) imagery in [0, 1].
    """

    def __init__(
        self,
        brightness_threshold: float = 0.65,
        whiteness_threshold: float = 0.10,
        dilate_pixels: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        brightness_threshold : Mean reflectance above which a pixel is cloud-suspect.
        whiteness_threshold  : Max band std deviation — clouds are spectrally flat.
        dilate_pixels        : Morphological dilation radius.
        """
        self.brightness_threshold = brightness_threshold
        self.whiteness_threshold = whiteness_threshold
        self.dilate_pixels = dilate_pixels

    def generate(self, image: np.ndarray) -> CloudMaskResult:
        """
        Parameters
        ----------
        image : [C, H, W] float32 in [0, 1]. Uses first 3 bands as RGB.

        Returns
        -------
        CloudMaskResult
        """
        if image.shape[0] >= 3:
            rgb = image[:3]  # [3, H, W]
        else:
            rgb = np.repeat(image[:1], 3, axis=0)

        # Brightness: mean of bands
        brightness = np.mean(rgb, axis=0)         # [H, W]

        # Whiteness: std across bands (low = spectrally flat = cloud)
        whiteness = np.std(rgb, axis=0)           # [H, W]

        # Cloud pixels: bright AND spectrally flat
        mask = (
            (brightness > self.brightness_threshold) &
            (whiteness < self.whiteness_threshold)
        ).astype(np.uint8)

        if self.dilate_pixels > 0:
            mask = SCLCloudDetector._dilate(mask, self.dilate_pixels)

        cloud_fraction = float(np.mean(mask))
        return CloudMaskResult(
            mask=mask,
            cloud_fraction=cloud_fraction,
            method="threshold",
            confidence=0.70,  # Lower confidence — heuristic method
        )


def auto_detect_clouds(
    image: np.ndarray,
    scl: np.ndarray | None = None,
    include_shadow: bool = True,
) -> CloudMaskResult:
    """
    Automatically select the best cloud detector.

    Uses SCL if available, falls back to threshold detection.
    """
    if scl is not None:
        return SCLCloudDetector(include_shadow=include_shadow).generate(scl)
    return ThresholdCloudDetector().generate(image)
