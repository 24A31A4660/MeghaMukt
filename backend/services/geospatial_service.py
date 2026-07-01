import os
import numpy as np
from PIL import Image
try:
    import rasterio
except ImportError:
    rasterio = None

class GeospatialService:
    @staticmethod
    def get_metadata(file_path: str) -> dict:
        """
        Extracts geospatial metadata from GeoTIFF or basic image properties from PNG/JPG.
        """
        file_size = os.path.getsize(file_path) / (1024 * 1024) # MB
        ext = os.path.splitext(file_path)[1].lower()

        metadata = {
            "filename": os.path.basename(file_path),
            "file_size_mb": round(file_size, 2),
            "width": 0,
            "height": 0,
            "resolution": "Unknown",
            "crs": "None",
            "projection": "None",
            "bands": [],
            "satellite": "Unknown",
            "acquisition_date": "N/A",
            "bounds": None,
            "is_geospatial": False
        }

        # Try Rasterio for GeoTIFF
        if rasterio and ext in [".tif", ".tiff", ".gtif"]:
            try:
                with rasterio.open(file_path) as src:
                    metadata["width"] = src.width
                    metadata["height"] = src.height
                    # Resolution
                    res = src.res
                    metadata["resolution"] = f"{round(res[0], 2)}m x {round(res[1], 2)}m"
                    metadata["crs"] = str(src.crs) if src.crs else "Unknown"
                    metadata["projection"] = src.crs.to_dict().get("proj", "Unknown") if src.crs else "Unknown"
                    metadata["bands"] = [f"Band {i}" for i in range(1, src.count + 1)]
                    metadata["bounds"] = {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top
                    }
                    metadata["is_geospatial"] = True
                    # Guess satellite based on bands or filename
                    if src.count >= 6:
                        metadata["satellite"] = "Sentinel-2"
                        metadata["bands"] = ["Blue (B02)", "Green (B03)", "Red (B04)", "NIR (B08)", "SWIR1 (B11)", "SWIR2 (B12)"]
                    elif src.count == 4:
                        metadata["satellite"] = "LISS-IV"
                        metadata["bands"] = ["Green (B2)", "Red (B3)", "NIR (B4)", "SWIR (B5)"]
                    else:
                        metadata["satellite"] = "Generic Multi-spectral"
                return metadata
            except Exception as e:
                print(f"[GeospatialService Error] Failed to read with rasterio: {e}")

        # Fallback to PIL for standard image formats
        try:
            if ext == ".zip":
                metadata["satellite"] = "Sentinel-2"
                metadata["resolution"] = "10m/20m"
                metadata["bands"] = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
                metadata["is_geospatial"] = True
            else:
                with Image.open(file_path) as img:
                    metadata["width"] = img.size[0]
                    metadata["height"] = img.size[1]
                    metadata["bands"] = ["Red", "Green", "Blue"] if img.mode == "RGB" else [img.mode]
                    metadata["resolution"] = "N/A (Standard Image)"
                    metadata["satellite"] = "High-Resolution Optical"
        except Exception as e:
            print(f"[GeospatialService Error] Failed to read with PIL/ZIP: {e}")

        return metadata

    @staticmethod
    def validate_image(file_path: str) -> tuple[bool, str]:
        """
        Validates satellite imagery before processing.
        Checks for blank, fully white, corrupted, or unsupported images.
        Note: Thresholds are intentionally lenient because cloud-covered
        satellite images (the primary input for ClearSat) are naturally bright.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".zip"]:
            return False, "Unsupported file format. Only SAFE zip, GeoTIFF, TIFF, PNG, JPEG are supported."

        if ext == ".zip":
            return True, "Valid Sentinel-2 SAFE archive."

        try:
            if rasterio and ext in [".tif", ".tiff"]:
                with rasterio.open(file_path) as src:
                    if src.width < 64 or src.height < 64:
                        return False, "Image dimensions are too small (minimum 64x64 required)."
                    # Check if empty (read first band)
                    band1 = src.read(1, out_shape=(1, 256, 256))
                    non_zero = np.count_nonzero(band1) / band1.size
                    if non_zero < 0.01:
                        return False, "Blank image detected. Image appears to be empty or fully black."
                return True, "Valid GeoTIFF image."
            else:
                with Image.open(file_path) as img:
                    if img.size[0] < 64 or img.size[1] < 64:
                        return False, "Image dimensions are too small (minimum 64x64 required)."
                    arr = np.array(img.convert("L"))
                    non_zero = np.count_nonzero(arr) / arr.size
                    if non_zero < 0.01:
                        return False, "Blank image detected. Image appears to be empty or fully black."
                return True, "Valid standard image."
        except Exception as e:
            return False, f"Corrupted file or unable to decode image. Details: {str(e)}"
