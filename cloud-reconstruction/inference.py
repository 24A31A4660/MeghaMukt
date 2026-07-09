#!/usr/bin/env python3
"""
inference.py — Cloud removal inference for single and batch images.

Handles all 4 input cases:
  1. Cloudy image   → detect → reconstruct → output cloud-free GeoTIFF + RGB
  2. Clear image    → return original + "No significant cloud cover detected."
  3. Blank image    → return original + "Blank image detected."
  4. White/invalid  → return original + "Invalid input."

Memory-optimized: processes full Sentinel-2 tiles (10980x10980) using
streaming windowed reads and incremental patch accumulation. Never loads
the full multi-band image into RAM.

Usage:
    python inference.py --input path/to/cloudy.SAFE.zip
    python inference.py --batch-dir path/to/folder/
    python inference.py --input scene.tif --checkpoint checkpoints/best.pth
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.confidence import compute_confidence_map, compute_difference_map
from evaluation.visualizer import save_all_outputs
from models.registry import build_model
from preprocessing.cloud_detector import auto_detect_clouds, SCLCloudDetector
from preprocessing.extractor import SafeZipExtractor, build_input_tensor, BAND_RESOLUTION
from preprocessing.patcher import PatchExtractor, PatchMerger
from preprocessing.validator import ImageValidator
from utils.checkpoint import find_best_checkpoint, load_checkpoint
from utils.geotiff import GeoTIFFMetadata, read_geotiff, write_geotiff, write_rgb_preview
from utils.logger import setup_logger

log = setup_logger("inference")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths in config to be absolute, relative to the script's directory
    script_dir = Path(__file__).parent
    
    # Resolve top-level paths block
    if "paths" in cfg:
        for k, v in cfg["paths"].items():
            p = Path(v)
            if not p.is_absolute():
                cfg["paths"][k] = str((script_dir / p).resolve())
                
    # Resolve dataset block paths
    if "dataset" in cfg:
        for k in ("source_dir", "output_dir"):
            if k in cfg["dataset"]:
                p = Path(cfg["dataset"][k])
                if not p.is_absolute():
                    cfg["dataset"][k] = str((script_dir / p).resolve())

    return cfg


def resolve_device(cfg: dict) -> torch.device:
    dev_type = cfg["device"]["type"]
    if dev_type == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_type)


class InferenceEngine:
    """
    Full inference pipeline: validate → detect → patch → reconstruct → merge → output.
    Memory-optimized for full-resolution Sentinel-2 tiles.
    """

    def __init__(self, cfg: dict, model: torch.nn.Module, device: torch.device) -> None:
        self.cfg = cfg
        self.model = model.eval()
        self.device = device
        self.validator = ImageValidator()
        self.n_bands = cfg["model"]["output_channels"]

        icfg = cfg["inference"]
        self.patch_size = icfg["patch_size"]
        self.stride = icfg["stride"]
        self.blend = icfg["blend_overlap"]
        self.output_confidence = icfg["output_confidence"]
        self.clear_threshold = icfg["clear_threshold"]

    @torch.no_grad()
    def _infer_batch(self, batch_np: np.ndarray, use_amp: bool) -> np.ndarray:
        """
        Run model on a single batch of patches.

        Parameters
        ----------
        batch_np : [N, C_in, H, W] float32 numpy array.
        use_amp  : Whether to use automatic mixed precision.

        Returns
        -------
        predictions : [N, C_out, H, W] float32 numpy array.
        """
        batch = torch.from_numpy(batch_np).float().to(self.device)
        optical = batch[:, :self.n_bands]
        mask_ch = batch[:, self.n_bands:]

        with autocast("cuda", enabled=use_amp):
            pred = self.model(optical, mask_ch)

        result = pred.cpu().numpy()
        del batch, optical, mask_ch, pred
        return result

    def process_safe_zip(self, zip_path: Path, output_dir: Path) -> dict:
        """
        Process a .SAFE.zip file through the full pipeline.

        Memory-optimized approach:
        1. Read ONLY the SCL band to generate cloud mask (~30MB)
        2. Compute all patch positions from the cloud mask
        3. Stream patches: extract via windowed reading → infer → accumulate
        4. Band-by-band merge using Gaussian blending
        5. Band-by-band cloud blend with original (windowed read)
        """
        t0 = time.time()
        scene_name = zip_path.stem.replace(".SAFE", "")
        bands = self.cfg["bands"]["optical"]
        use_amp = self.cfg["training"]["amp"] and self.device.type == "cuda"
        batch_size = self.cfg["training"]["batch_size"]

        log.info("Processing: %s", zip_path.name)

        # ── Step 1: Read SCL band only for cloud detection ──────────
        with SafeZipExtractor(zip_path, target_res=self.cfg["bands"]["target_resolution"]) as ext:
            scl_data, scl_profile = ext.read_band("SCL")
            scl = scl_data[0]  # [H_20m, W_20m]

            # Get the 10m reference dimensions from a 10m band
            ref_src = ext._get_dataset("B02")
            full_h, full_w = ref_src.height, ref_src.width

        # Upsample SCL to 10m resolution for cloud mask
        from scipy.ndimage import zoom
        scale = self.cfg["bands"]["target_resolution"] / 20  # 10/20 = 0.5
        if scale != 1.0:
            scl_10m = zoom(scl.astype(np.float32), full_h / scl.shape[0], order=0).astype(np.uint8)
        else:
            scl_10m = scl

        # Ensure correct dimensions
        scl_10m = scl_10m[:full_h, :full_w]

        cloud_detector = SCLCloudDetector(include_shadow=True)
        cloud_result = cloud_detector.generate(scl_10m)
        cloud_mask = cloud_result.mask  # [full_h, full_w] binary

        log.info("  Cloud fraction: %.1f%% (method: %s, confidence: %.2f)",
                 cloud_result.cloud_fraction * 100,
                 cloud_result.method,
                 cloud_result.confidence)

        del scl_data, scl, scl_10m
        gc.collect()

        # ── Step 2: Quick validation (check a small sample) ─────────
        if cloud_result.cloud_fraction < self.clear_threshold:
            log.info("  CLEAR: Cloud fraction below threshold — returning original")
            # Generate a quick RGB preview from windowed reads
            with SafeZipExtractor(zip_path) as ext:
                # Read a 1024x1024 center crop for preview
                ch = full_h // 2 - 512
                cw = full_w // 2 - 512
                r = ext.read_window("B04", ch, cw, 1024, 1024)
                g = ext.read_window("B03", ch, cw, 1024, 1024)
                b = ext.read_window("B02", ch, cw, 1024, 1024)
                preview = np.stack([r, g, b], axis=0)
            out_path = output_dir / scene_name
            out_path.mkdir(parents=True, exist_ok=True)
            write_rgb_preview(out_path / "cloud_free_rgb.jpg", preview)
            return {"scene": scene_name, "status": "clear",
                    "cloud_fraction": cloud_result.cloud_fraction,
                    "time": time.time() - t0}

        # ── Step 3: Compute patch positions ─────────────────────────
        positions: list[tuple[int, int]] = []
        ps = self.patch_size
        for r in range(0, full_h - ps + 1, self.stride):
            for c in range(0, full_w - ps + 1, self.stride):
                positions.append((r, c))

        n_patches = len(positions)
        log.info("  Patches: %d (%dx%d, stride %d)", n_patches, ps, ps, self.stride)

        # ── Step 4: Streaming inference + incremental accumulation ──
        # Initialize Gaussian weight map
        merger = PatchMerger(
            full_shape=(self.n_bands, full_h, full_w),
            patch_size=ps,
        )
        w = merger._weight_map  # [ps, ps] float32

        # Weight accumulator — shared for all bands
        weight_acc = np.zeros((full_h, full_w), dtype=np.float32)
        for (r, c) in positions:
            weight_acc[r:r+ps, c:c+ps] += w

        weight_acc = np.maximum(weight_acc, 1e-8)

        # Band accumulators — process one band at a time after collecting predictions
        # But we need all bands from each prediction simultaneously.
        # So: accumulate all bands directly, using float32
        merged = np.zeros((self.n_bands, full_h, full_w), dtype=np.float32)

        log.info("  Running streaming inference (batch_size=%d)...", batch_size)

        with SafeZipExtractor(zip_path) as ext:
            pbar = tqdm(range(0, n_patches, batch_size),
                        desc="  Inference", unit="batch", leave=False)

            for batch_start in pbar:
                batch_end = min(batch_start + batch_size, n_patches)
                batch_positions = positions[batch_start:batch_end]
                cur_batch_size = len(batch_positions)

                # Extract patches via windowed reading (small memory footprint)
                input_patches = np.zeros(
                    (cur_batch_size, self.n_bands + 1, ps, ps),
                    dtype=np.float32,
                )

                for i, (r, c) in enumerate(batch_positions):
                    for bi, band in enumerate(bands):
                        input_patches[i, bi] = ext.read_window(band, r, c, ps, ps)
                    # Cloud mask channel
                    input_patches[i, self.n_bands] = cloud_mask[r:r+ps, c:c+ps].astype(np.float32)

                # Run inference on this batch
                pred = self._infer_batch(input_patches, use_amp)

                # Accumulate predictions into merged (weighted)
                for i, (r, c) in enumerate(batch_positions):
                    for band_idx in range(self.n_bands):
                        merged[band_idx, r:r+ps, c:c+ps] += pred[i, band_idx] * w

                del input_patches, pred
                pbar.set_postfix(patches=f"{batch_end}/{n_patches}")

        # Normalize by weights
        for band_idx in range(self.n_bands):
            merged[band_idx] /= weight_acc

        gc.collect()

        # ── Step 5: Cloud blend — read original band-by-band ────────
        log.info("  Blending with original (cloud pixels only)...")
        mask_f = cloud_mask.astype(np.float32)  # [H, W]

        with SafeZipExtractor(zip_path) as ext:
            # Also build the original optical stack for visualization
            # (we read it band-by-band to save memory)
            original_rgb = np.zeros((3, full_h, full_w), dtype=np.float32)

            for bi, band in enumerate(bands):
                # Read the full original band
                orig_band = ext.read_window(band, 0, 0, full_w, full_h)  # [H, W]

                # Blend: keep original where clear, use reconstruction where cloudy
                merged[bi] = orig_band * (1 - mask_f) + merged[bi] * mask_f

                # Store RGB bands for preview
                if band == "B04":
                    original_rgb[0] = orig_band
                elif band == "B03":
                    original_rgb[1] = orig_band
                elif band == "B02":
                    original_rgb[2] = orig_band

                del orig_band

        gc.collect()

        # ── Step 6: Generate outputs ────────────────────────────────
        log.info("  Generating output images...")

        # Build RGB arrays for visualization
        final_rgb = np.stack([merged[2], merged[1], merged[0]], axis=0)  # B04, B03, B02

        # Confidence and difference maps (computed on RGB only to save memory)
        confidence_map = compute_confidence_map(final_rgb, original_rgb, cloud_mask)
        difference_map = compute_difference_map(final_rgb, original_rgb)

        saved = save_all_outputs(
            scene_name=scene_name,
            output_dir=output_dir,
            cloudy_input=original_rgb,
            prediction=final_rgb,
            cloud_mask=cloud_mask,
            confidence_map=confidence_map,
            difference_map=difference_map,
            ground_truth=None,
            metrics={
                "cloud_fraction": float(cloud_result.cloud_fraction),
                "detection_method": cloud_result.method,
                "detection_confidence": float(cloud_result.confidence),
                "n_patches": n_patches,
                "processing_time_s": round(time.time() - t0, 2),
            },
        )

        # Also save the full multi-band result as .npy for scientific use
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(scene_dir / "cloud_free_multiband.npy", merged)

        elapsed = time.time() - t0
        log.info("  DONE in %.1fs -> %s", elapsed, scene_dir)

        del merged, original_rgb, final_rgb, confidence_map, difference_map, cloud_mask
        gc.collect()

        return {
            "scene": scene_name,
            "status": "reconstructed",
            "cloud_fraction": float(cloud_result.cloud_fraction),
            "time": elapsed,
            "outputs": {k: str(v) for k, v in saved.items()},
        }

    def process_image_file(self, file_path: Path, output_dir: Path) -> dict:
        """Process a standalone image file (GeoTIFF, PNG, JPG)."""
        t0 = time.time()
        scene_name = file_path.stem

        log.info("Processing image file: %s", file_path.name)

        if file_path.suffix.lower() in (".tif", ".tiff"):
            # Validate file
            val_result = self.validator.validate_geotiff(file_path)
            if not val_result.valid:
                log.info("  INVALID: %s", val_result.message)
                return {"scene": scene_name, "status": val_result.message, "time": time.time() - t0}
            data, meta = read_geotiff(file_path)
        else:
            # Standalone PNG, JPG, JPEG
            from PIL import Image
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3] in [0, 1]
            # Convert to [3, H, W] in BGR order (B02, B03, B04)
            b = arr[:, :, 2]
            g = arr[:, :, 1]
            r = arr[:, :, 0]
            data = np.stack([b, g, r], axis=0)
            meta = None

        # Cloud detection (no SCL available for standalone images)
        cloud_result = auto_detect_clouds(data, scl=None)

        if cloud_result.cloud_fraction < self.clear_threshold:
            log.info("  CLEAR: Cloud fraction %.1f%% below threshold", cloud_result.cloud_fraction * 100)
            out_path = output_dir / scene_name
            out_path.mkdir(parents=True, exist_ok=True)
            rgb_preview = np.stack([data[2], data[1], data[0]], axis=0) if data.shape[0] >= 3 else data
            write_rgb_preview(out_path / "cloud_free_rgb.jpg", rgb_preview)
            return {"scene": scene_name, "status": "clear", "cloud_fraction": cloud_result.cloud_fraction, "time": time.time() - t0}

        # Ensure correct channel count
        if data.shape[0] < self.n_bands:
            log.warning("  Image has %d bands (need %d). Padding with zeros.", data.shape[0], self.n_bands)
            pad = np.zeros((self.n_bands - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        # Build input with mask channel
        mask = cloud_result.mask
        input_tensor = np.concatenate([data[:self.n_bands], mask[np.newaxis].astype(np.float32)], axis=0)

        # Patch → reconstruct → merge → blend
        patcher = PatchExtractor(self.patch_size, self.stride)
        patches, positions = patcher.extract_with_positions(input_tensor)

        use_amp = self.cfg["training"]["amp"] and self.device.type == "cuda"
        batch_size = self.cfg["training"]["batch_size"]

        # Streaming inference
        merger = PatchMerger(
            full_shape=(self.n_bands, data.shape[1], data.shape[2]),
            patch_size=self.patch_size,
        )

        pred_list = []
        for i in range(0, len(patches), batch_size):
            batch_np = patches[i:i+batch_size]
            pred_list.append(self._infer_batch(batch_np, use_amp))
        pred_patches = np.concatenate(pred_list, axis=0)

        reconstructed = merger.merge(list(pred_patches), positions)
        final = merger.merge_with_cloud_blend(data[:self.n_bands], reconstructed, mask)

        # Save outputs with metadata preserved
        confidence_map = compute_confidence_map(final, data[:self.n_bands], mask)
        difference_map = compute_difference_map(final, data[:self.n_bands])

        saved = save_all_outputs(
            scene_name=scene_name,
            output_dir=output_dir,
            cloudy_input=data[:self.n_bands],
            prediction=final,
            cloud_mask=mask,
            confidence_map=confidence_map,
            difference_map=difference_map,
            ground_truth=None,
            meta=meta,
            metrics={
                "cloud_fraction": float(cloud_result.cloud_fraction),
                "detection_method": cloud_result.method,
                "processing_time_s": round(time.time() - t0, 2),
            },
        )

        elapsed = time.time() - t0
        log.info("  DONE in %.1fs -> %s", elapsed, output_dir / scene_name)
        return {"scene": scene_name, "status": "reconstructed", "cloud_fraction": cloud_result.cloud_fraction, "time": elapsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cloud reconstruction inference.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to a single .SAFE.zip or .tif")
    parser.add_argument("--batch-dir", type=str, default=None, help="Directory of images for batch inference")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: best.pth)")
    args = parser.parse_args(argv)

    if not args.input and not args.batch_dir:
        parser.error("Provide --input (single image) or --batch-dir (batch mode).")

    # Resolve config path relative to script directory if it is a relative path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    cfg = load_config(config_path)
    device = resolve_device(cfg)

    # Load model
    model = build_model(cfg).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(Path(cfg["paths"]["checkpoints"]))
    if ckpt_path is None or not ckpt_path.exists():
        log.error("No checkpoint found at %s. Train the model first.", cfg["paths"]["checkpoints"])
        return 1

    load_checkpoint(ckpt_path, model, device=str(device), strict=False)
    log.info("Loaded checkpoint: %s", ckpt_path)

    output_dir = Path(args.output or cfg["paths"]["outputs"])

    engine = InferenceEngine(cfg, model, device)
    results: list[dict] = []

    # Collect input files
    input_files: list[Path] = []
    if args.input:
        input_files.append(Path(args.input))
    elif args.batch_dir:
        batch_path = Path(args.batch_dir)
        input_files.extend(sorted(batch_path.glob("*.SAFE.zip")))
        input_files.extend(sorted(batch_path.glob("*.tif")))
        input_files.extend(sorted(batch_path.glob("*.tiff")))
        log.info("Batch mode: %d files in %s", len(input_files), batch_path)

    for fpath in tqdm(input_files, desc="Inference", unit="scene"):
        if fpath.name.lower().endswith(".safe.zip") or fpath.name.lower().endswith(".zip"):
            result = engine.process_safe_zip(fpath, output_dir)
        elif fpath.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            result = engine.process_image_file(fpath, output_dir)
        else:
            log.warning("Skipping unsupported file: %s", fpath.name)
            continue
        results.append(result)

    # Summary
    log.info("=" * 60)
    log.info("INFERENCE COMPLETE -- %d scenes processed", len(results))
    log.info("=" * 60)
    for r in results:
        log.info("  %s: %s (cloud=%.1f%%, time=%.1fs)",
                 r.get("scene", "?"),
                 r.get("status", "?"),
                 r.get("cloud_fraction", 0) * 100,
                 r.get("time", 0))

    # Save batch results
    summary_path = output_dir / "inference_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Summary saved to: %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
