"""preprocessing/extractor.py — Extract spectral bands from Sentinel-2 .SAFE.zip files."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional, Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

# Band → native resolution inside .SAFE.zip
BAND_RESOLUTION: dict[str, int] = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B11": 20, "B12": 20,
    "SCL": 20,
}


class SafeZipExtractor:
    """
    Extracts spectral bands and SCL from a Sentinel-2 L2A .SAFE.zip.
    Optimized for memory efficiency by supporting windowed reading.
    """

    def __init__(
        self,
        zip_path: Path,
        target_res: int = 10,
        resampling: Resampling = Resampling.bilinear,
    ) -> None:
        self.zip_path = Path(zip_path)
        self.target_res = target_res
        self.resampling = resampling
        self._name_cache: Optional[list[str]] = None
        self._open_memfiles: dict[str, Any] = {}
        self._open_datasets: dict[str, Any] = {}
        self._zf: Optional[zipfile.ZipFile] = None

    def __enter__(self) -> SafeZipExtractor:
        self._zf = zipfile.ZipFile(self.zip_path, "r")
        self._name_cache = self._zf.namelist()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close all open datasets, memory files, and the zip file."""
        for src in self._open_datasets.values():
            try:
                src.close()
            except Exception:
                pass
        self._open_datasets.clear()

        for memfile in self._open_memfiles.values():
            try:
                memfile.close()
            except Exception:
                pass
        self._open_memfiles.clear()

        if self._zf is not None:
            try:
                self._zf.close()
            except Exception:
                pass
            self._zf = None

    def _get_names(self) -> list[str]:
        if self._name_cache is None:
            if self._zf is not None:
                self._name_cache = self._zf.namelist()
            else:
                with zipfile.ZipFile(self.zip_path, "r") as zf:
                    self._name_cache = zf.namelist()
        return self._name_cache

    def _find_band_entry(self, band: str, prefer_res: int) -> Optional[str]:
        """Find the zip entry path for a given band at the preferred resolution."""
        names = self._get_names()
        candidates: list[tuple[int, str]] = []
        for name in names:
            fname = name.split("/")[-1]
            m = re.search(rf"_{band}_(\d+)m\.jp2$", fname, re.IGNORECASE)
            if m:
                res = int(m.group(1))
                candidates.append((res, name))

        if not candidates:
            return None

        for res, name in sorted(candidates, key=lambda x: abs(x[0] - prefer_res)):
            return name
        return None

    def _get_dataset(self, band: str) -> rasterio.DatasetReader:
        """Get or open the rasterio dataset for a given band, keeping it cached."""
        band = band.upper()
        if band in self._open_datasets:
            return self._open_datasets[band]

        native_res = BAND_RESOLUTION.get(band, 10)
        entry = self._find_band_entry(band, native_res)
        if entry is None:
            raise FileNotFoundError(f"Band {band} not found in {self.zip_path.name}")

        # Read bytes from zip
        if self._zf is not None:
            raw = self._zf.read(entry)
        else:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                raw = zf.read(entry)

        # Create memory file and open dataset
        memfile = rasterio.MemoryFile(raw)
        src = memfile.open()

        self._open_memfiles[band] = memfile
        self._open_datasets[band] = src
        return src

    def read_band(self, band: str) -> tuple[np.ndarray, rasterio.profiles.Profile]:
        """Read a single band at native resolution."""
        src = self._get_dataset(band)
        data = src.read()
        return data, src.profile.copy()

    def read_window(
        self,
        band: str,
        row_10m: int,
        col_10m: int,
        width_10m: int,
        height_10m: int,
    ) -> np.ndarray:
        """
        Read a window of a band, automatically handling resolution differences.
        Values are returned as float32 in [0, 1] for spectral bands, and uint8 for SCL.
        """
        band = band.upper()
        src = self._get_dataset(band)
        native_res = BAND_RESOLUTION.get(band, 10)

        if native_res == 10:
            # Native 10m band
            window = Window(col_10m, row_10m, width_10m, height_10m)
            data = src.read(1, window=window)
        else:
            # 20m band (B11, B12, SCL)
            # Scale coordinates and size to 20m
            row_20m = row_10m // 2
            col_20m = col_10m // 2
            width_20m = width_10m // 2
            height_20m = height_10m // 2

            window = Window(col_20m, row_20m, width_20m, height_20m)

            if band == "SCL":
                # Nearest neighbor resampling for classification labels
                data = src.read(
                    1,
                    window=window,
                    out_shape=(height_10m, width_10m),
                    resampling=Resampling.nearest,
                )
            else:
                # Bilinear resampling for spectral bands
                data = src.read(
                    1,
                    window=window,
                    out_shape=(height_10m, width_10m),
                    resampling=self.resampling,
                )

        arr = data.astype(np.float32)
        if band != "SCL":
            arr = np.clip(arr / 10000.0, 0.0, 1.0)
        else:
            arr = arr.astype(np.uint8)

        return arr

    def extract_bands(
        self,
        bands: list[str],
        include_scl: bool = True,
    ) -> dict[str, np.ndarray]:
        """Legacy method: extract full bands. Warning: high memory usage."""
        # Find first 10m band to get shape
        ref_band = next((b for b in bands if BAND_RESOLUTION.get(b.upper(), 10) == self.target_res), bands[0])
        ref_src = self._get_dataset(ref_band)
        target_h, target_w = ref_src.height, ref_src.width

        result: dict[str, np.ndarray] = {}
        for band in bands:
            result[band] = self.read_window(band, 0, 0, target_w, target_h)

        if include_scl:
            try:
                result["SCL"] = self.read_window("SCL", 0, 0, target_w, target_h)
            except Exception:
                pass

        return result

    def get_tile_id(self) -> str:
        """Extract tile ID (e.g. T44QND) from the zip filename."""
        m = re.search(r"T\d{2}[A-Z]{3}", self.zip_path.name)
        return m.group() if m else "UNKNOWN"

    def get_acquisition_date(self) -> str:
        """Extract acquisition date string YYYYMMDD from filename."""
        m = re.search(r"_(\d{8})T\d{6}_", self.zip_path.name)
        return m.group(1) if m else "unknown"


def build_input_tensor(
    bands_dict: dict[str, np.ndarray],
    band_order: list[str],
    cloud_mask: np.ndarray,
) -> np.ndarray:
    """Stack optical bands + cloud mask into a single input tensor."""
    bands = [bands_dict[b] for b in band_order]
    bands.append(cloud_mask.astype(np.float32))
    return np.stack(bands, axis=0)  # [C, H, W]
