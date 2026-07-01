"""preprocessing/validator.py — Image validation before AI processing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ValidationResult:
    valid: bool
    message: str
    cloud_fraction: float = 0.0
    image_type: str = "unknown"   # "cloudy" | "clear" | "blank" | "invalid"


class ImageValidator:
    """
    Validates satellite imagery before processing.

    Checks performed:
    - File existence and readability
    - Supported format (GeoTIFF, JP2)
    - Not fully black (empty / sensor failure)
    - Not fully white (saturated / invalid)
    - Sufficient spatial extent (not tiny)
    - Valid data range
    """

    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jp2", ".img"}
    MIN_SIZE_PX = 64          # Minimum width/height in pixels
    BLACK_THRESHOLD = 0.01    # Fraction of non-zero pixels below which image is "blank"
    WHITE_THRESHOLD = 0.95    # Fraction of saturated pixels above which image is "white"
    CLEAR_CLOUD_THRESHOLD = 0.05   # Cloud fraction below which image is "clear"

    def validate_path(self, path: Path) -> ValidationResult:
        """Check file exists and has a supported extension."""
        path = Path(path)
        if not path.exists():
            return ValidationResult(False, f"File not found: {path.name}", image_type="invalid")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return ValidationResult(
                False,
                f"Unsupported format '{path.suffix}'. Supported: {self.SUPPORTED_EXTENSIONS}",
                image_type="invalid",
            )
        return ValidationResult(True, "Path OK")

    def validate_array(
        self,
        data: np.ndarray,
        cloud_mask: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        Validate a numpy array [C, H, W] or [H, W].

        Parameters
        ----------
        data        : Image array, values expected in [0, 1].
        cloud_mask  : Optional binary cloud mask [H, W].
        """
        if data.ndim == 2:
            data = data[np.newaxis]

        _, h, w = data.shape

        # Size check
        if h < self.MIN_SIZE_PX or w < self.MIN_SIZE_PX:
            return ValidationResult(
                False,
                f"Image too small ({w}x{h} px). Minimum: {self.MIN_SIZE_PX}px.",
                image_type="invalid",
            )

        # Empty / fully black
        non_zero = np.count_nonzero(data) / data.size
        if non_zero < self.BLACK_THRESHOLD:
            return ValidationResult(
                False,
                "Blank image detected. Image appears to be empty or fully black.",
                image_type="blank",
            )

        # Fully white / saturated
        # Treat values > 0.95 of max as saturated
        saturated = np.mean(data > 0.95)
        if saturated > self.WHITE_THRESHOLD:
            return ValidationResult(
                False,
                "Invalid input. Image appears fully saturated (all white).",
                image_type="invalid",
            )

        # Cloud fraction
        cloud_fraction = 0.0
        if cloud_mask is not None:
            cloud_fraction = float(np.mean(cloud_mask > 0))

        # Classify
        if cloud_fraction < self.CLEAR_CLOUD_THRESHOLD:
            return ValidationResult(
                True,
                "No significant cloud cover detected.",
                cloud_fraction=cloud_fraction,
                image_type="clear",
            )

        return ValidationResult(
            True,
            f"Valid cloudy image. Cloud cover: {cloud_fraction:.1%}",
            cloud_fraction=cloud_fraction,
            image_type="cloudy",
        )

    def validate_geotiff(self, path: Path) -> ValidationResult:
        """Full validation: path + open + array check."""
        # 1. Path check
        path_result = self.validate_path(path)
        if not path_result.valid:
            return path_result

        # 2. Try to open
        try:
            import rasterio
            with rasterio.open(path) as src:
                if src.count == 0:
                    return ValidationResult(False, "GeoTIFF has 0 bands.", image_type="invalid")
                # Read a small overview to avoid loading full tile
                overview_data = src.read(
                    out_shape=(src.count, min(256, src.height), min(256, src.width))
                ).astype(np.float32) / 10000.0
        except Exception as exc:
            return ValidationResult(
                False, f"Cannot read GeoTIFF: {exc}", image_type="invalid"
            )

        # 3. Array checks
        return self.validate_array(overview_data)
