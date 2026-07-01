"""utils/geotiff.py — GeoTIFF I/O with full metadata preservation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine


class GeoTIFFMetadata:
    """Container for geospatial metadata extracted from a rasterio dataset."""

    def __init__(self, src: rasterio.DatasetReader) -> None:
        self.crs: Optional[CRS] = src.crs
        self.transform: Affine = src.transform
        self.width: int = src.width
        self.height: int = src.height
        self.count: int = src.count
        self.dtype: str = src.dtypes[0]
        self.nodata: Optional[float] = src.nodata
        self.tags: dict = src.tags()
        self.res: tuple[float, float] = src.res          # (pixel_width, pixel_height)
        self.bounds = src.bounds
        self.profile: dict = src.profile.copy()

    def __repr__(self) -> str:
        return (
            f"GeoTIFFMetadata(crs={self.crs}, res={self.res}, "
            f"size={self.width}x{self.height}, bands={self.count})"
        )


def read_geotiff(
    path: Path,
    bands: Optional[list[int]] = None,
    resample_to: Optional[tuple[int, int]] = None,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, GeoTIFFMetadata]:
    """
    Read a GeoTIFF file.

    Parameters
    ----------
    path        : Path to GeoTIFF.
    bands       : 1-based band indices to read (None = all).
    resample_to : Optional (height, width) to resample the output to.
    resampling  : Rasterio resampling method.

    Returns
    -------
    data  : ndarray [C, H, W] float32, normalised to [0, 1].
    meta  : GeoTIFFMetadata
    """
    with rasterio.open(path) as src:
        meta = GeoTIFFMetadata(src)
        band_indices = bands if bands else list(range(1, src.count + 1))

        if resample_to:
            out_h, out_w = resample_to
            data = src.read(
                band_indices,
                out_shape=(len(band_indices), out_h, out_w),
                resampling=resampling,
            )
        else:
            data = src.read(band_indices)

    data = data.astype(np.float32)
    # Normalise reflectance (Sentinel-2 L2A: uint16, max ~10000)
    data = np.clip(data / 10000.0, 0.0, 1.0)
    return data, meta


def write_geotiff(
    path: Path,
    data: np.ndarray,
    meta: GeoTIFFMetadata,
    band_descriptions: Optional[list[str]] = None,
    extra_tags: Optional[dict] = None,
    compress: str = "lzw",
) -> Path:
    """
    Write a multi-band GeoTIFF preserving all geospatial metadata.

    Parameters
    ----------
    path              : Output file path.
    data              : ndarray [C, H, W] float32, values in [0, 1].
    meta              : GeoTIFFMetadata from the source image.
    band_descriptions : Optional list of band names.
    extra_tags        : Additional tags to embed.
    compress          : Compression codec.

    Returns
    -------
    path : The written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis]  # [H,W] -> [1,H,W]

    n_bands, height, width = data.shape

    # Scale back to uint16 (Sentinel-2 convention)
    out_data = np.clip(data * 10000.0, 0, 65535).astype(np.uint16)

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": n_bands,
        "crs": meta.crs,
        "transform": meta.transform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": None,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out_data)
        # Preserve original tags + add new ones
        tags = dict(meta.tags)
        if extra_tags:
            tags.update(extra_tags)
        tags["CLOUD_RECONSTRUCTION"] = "true"
        dst.update_tags(**tags)

        if band_descriptions:
            for i, desc in enumerate(band_descriptions[:n_bands], start=1):
                dst.update_tags(i, description=desc)

    return path


def write_rgb_preview(
    path: Path,
    data: np.ndarray,
    quality: int = 90,
) -> Path:
    """
    Save an RGB preview from a [C, H, W] float32 array (values in [0, 1]).
    Saves as PNG if path ends with .png, JPEG otherwise.
    Handles 3-band RGB and 6-band Sentinel-2 (B,G,R,NIR,SWIR1,SWIR2) automatically.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 3:
        n_bands = data.shape[0]
        if n_bands == 3:
            # Standard RGB — keep order as-is
            rgb = np.stack([data[0], data[1], data[2]], axis=-1)  # [H,W,3]
        elif n_bands >= 6:
            # Sentinel-2: B02=Blue(0), B03=Green(1), B04=Red(2) → display as R,G,B
            rgb = np.stack([data[2], data[1], data[0]], axis=-1)  # [H,W,3]
        else:
            rgb = np.transpose(data[:3], (1, 2, 0))
    elif data.ndim == 2:
        rgb = np.stack([data, data, data], axis=-1)
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # Percentile contrast stretch for visualisation
    p2 = float(np.percentile(rgb, 2))
    p98 = float(np.percentile(rgb, 98))
    rgb = rgb.astype(np.float32)
    rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb_uint8)
    suffix = path.suffix.lower()
    if suffix == ".png":
        img.save(path, "PNG")
    else:
        img.save(path, "JPEG", quality=quality, optimize=True)
    return path

