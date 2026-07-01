#!/usr/bin/env python3
"""
prepare_dataset.py — Extract paired patches from .SAFE.zip files for training.
Optimized to use windowed reading to prevent memory errors.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.cloud_detector import SCLCloudDetector
from preprocessing.extractor import SafeZipExtractor
from utils.logger import setup_logger

log = setup_logger("prepare_dataset")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_scenes(source_dir: Path) -> dict[str, list[Path]]:
    scenes: dict[str, list[Path]] = {}
    for folder in ("clear", "medium", "cloudy"):
        d = source_dir / folder
        if d.exists():
            zips = sorted(d.glob("*.SAFE.zip"))
            if zips:
                scenes[folder] = zips
                log.info("Found %d scenes in %s/", len(zips), folder)
    return scenes


def group_by_tile(scenes: dict[str, list[Path]]) -> dict[str, dict[str, list[Path]]]:
    tiles: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for cls, paths in scenes.items():
        for p in paths:
            ext = SafeZipExtractor(p)
            tile_id = ext.get_tile_id()
            tiles[tile_id][cls].append(p)
    return dict(tiles)


def get_date(path: Path) -> datetime:
    m = re.search(r"_(\d{8})T\d{6}_", path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    return datetime.fromtimestamp(0)

def load_cloud_cover(source_dir: Path) -> dict[str, float]:
    csv_path = source_dir / "classification_report.csv"
    cloud_cover = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cloud_cover[row["Filename"].strip()] = float(row["Cloud Cover %"])
    return cloud_cover


def create_pairs(
    tiles: dict[str, dict[str, list[Path]]],
    source_dir: Path,
    output_dir: Path,
    include_medium: bool = True,
) -> list[tuple[Path, Path, str]]:
    cloud_cover = load_cloud_cover(source_dir)
    pairs = []
    rejected = []

    for tile_id, by_cls in tiles.items():
        clear_scenes = by_cls.get("clear", [])
        if not clear_scenes:
            log.warning("Tile %s has no clear reference — skipping", tile_id)
            for cloudy_path in by_cls.get("cloudy", []):
                rejected.append((cloudy_path.name, "None", "No clear reference in tile"))
            if include_medium:
                for medium_path in by_cls.get("medium", []):
                    rejected.append((medium_path.name, "None", "No clear reference in tile"))
            continue

        def find_closest_clear(target_path: Path) -> tuple[Path, int]:
            target_date = get_date(target_path)
            closest = min(clear_scenes, key=lambda p: abs((get_date(p) - target_date).days))
            delta = abs((get_date(closest) - target_date).days)
            return closest, delta

        candidates = by_cls.get("cloudy", [])
        if include_medium:
            candidates.extend(by_cls.get("medium", []))

        for target_path in candidates:
            ref_clear, delta = find_closest_clear(target_path)
            if delta > 30:
                rejected.append((target_path.name, ref_clear.name, f"> 30 days ({delta} days)"))
                log.info("REJECTED: %s (delta %d > 30)", target_path.name, delta)
            else:
                pairs.append((target_path, ref_clear, tile_id, delta))
                log.info("PAIR: %s <-> %s (delta %d)", target_path.name, ref_clear.name, delta)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pairing_report.md"
    
    report = ["# Pairing Report\n"]
    report.append(f"## Accepted Pairs ({len(pairs)})\n")
    for t, c, tid, d in pairs:
        report.append(f"- `{t.name}` <-> `{c.name}` (Δ {d} days)")
    
    report.append(f"\n## Rejected Pairs ({len(rejected)})\n")
    for t, c, reason in rejected:
        report.append(f"- `{t}` <-> `{c}` (Reason: {reason})")
    
    if pairs:
        deltas = [p[3] for p in pairs]
        avg_d = sum(deltas) / len(deltas)
        max_d = max(deltas)
        min_d = min(deltas)
        report.append(f"\n## Temporal Statistics\n")
        report.append(f"- **Average temporal difference:** {avg_d:.1f} days")
        report.append(f"- **Maximum temporal difference:** {max_d} days")
        report.append(f"- **Minimum temporal difference:** {min_d} days")
        
        # Cloud difference
        cloud_diffs = []
        for t, c, tid, d in pairs:
            c_pct = cloud_cover.get(t.name, 0.0)
            clr_pct = cloud_cover.get(c.name, 0.0)
            cloud_diffs.append(c_pct - clr_pct)
        if cloud_diffs:
            avg_c = sum(cloud_diffs) / len(cloud_diffs)
            report.append(f"- **Average cloud difference:** {avg_c:.1f}%")

    report_path.write_text("\n".join(report), encoding="utf-8")
    log.info("Wrote pairing report to %s", report_path.resolve())

    return [(p[0], p[1], p[2]) for p in pairs]


def extract_and_save_patches(
    pairs: list[tuple[Path, Path, str]],
    cfg: dict,
    output_dir: Path,
) -> dict[str, int]:
    pcfg = cfg["patch"]
    bands = cfg["bands"]["optical"]
    patch_size = pcfg["size"]
    stride = pcfg["stride"]
    min_cloud = pcfg["min_cloud_fraction"]
    max_cloud = pcfg["max_cloud_fraction"]

    train_r = cfg["dataset"]["train_ratio"]
    val_r   = cfg["dataset"]["val_ratio"]
    seed    = cfg["dataset"]["random_seed"]

    cloud_detector = SCLCloudDetector(include_shadow=True)

    # Create output directories
    for split in ("train", "validation", "test"):
        for sub in ("cloudy", "clear", "masks"):
            (output_dir / split / sub).mkdir(parents=True, exist_ok=True)

    all_patches: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    pair_idx = 0

    for cloudy_path, clear_path, tile_id in tqdm(pairs, desc="Extracting patches", unit="pair"):
        try:
            # 1. Read SCL first to determine cloud mask and find useful patch coordinates
            with SafeZipExtractor(cloudy_path, target_res=cfg["bands"]["target_resolution"]) as cloudy_ext:
                scl, _ = cloudy_ext.read_band("SCL")
                scl = scl[0]  # [H, W]

            cloud_result = cloud_detector.generate(scl)
            cloud_mask = cloud_result.mask
            h, w = cloud_mask.shape

            # Find coordinates of all overlapping patches
            candidate_positions = []
            for r in range(0, h - patch_size + 1, stride):
                for c in range(0, w - patch_size + 1, stride):
                    # Check cloud fraction of this crop on the mask
                    mask_crop = cloud_mask[r:r+patch_size, c:c+patch_size]
                    frac = float(np.mean(mask_crop))
                    if min_cloud <= frac <= max_cloud:
                        candidate_positions.append((r, c))

            if not candidate_positions:
                log.info("%s: No useful patches with cloud fraction in [%.2f, %.2f]",
                         tile_id, min_cloud, max_cloud)
                continue

            log.info("%s: Found %d useful patches. Extracting from zip...", tile_id, len(candidate_positions))

            # 2. Extract only the needed 256x256 windows from the zip files
            with SafeZipExtractor(cloudy_path) as cloudy_ext, SafeZipExtractor(clear_path) as clear_ext:
                for r, c in candidate_positions:
                    # Extract bands for this window
                    cloudy_patch_bands = []
                    clear_patch_bands = []

                    for band in bands:
                        c_band_patch = cloudy_ext.read_window(band, r, c, patch_size, patch_size)
                        t_band_patch = clear_ext.read_window(band, r, c, patch_size, patch_size)
                        cloudy_patch_bands.append(c_band_patch)
                        clear_patch_bands.append(t_band_patch)

                    # Extract mask patch
                    m_crop = cloud_mask[r:r+patch_size, c:c+patch_size]

                    # Stack bands
                    cloudy_patch_stack = np.stack(cloudy_patch_bands, axis=0)  # [C, H, W]
                    clear_patch_stack  = np.stack(clear_patch_bands, axis=0)   # [C, H, W]

                    # Add cloud mask as the 7th channel for the input
                    input_patch = np.concatenate(
                        [cloudy_patch_stack, m_crop[np.newaxis].astype(np.float32)],
                        axis=0
                    )

                    pair_name = f"pair_{pair_idx:06d}_{tile_id}_{r:05d}_{c:05d}"
                    all_patches.append((input_patch, clear_patch_stack, m_crop, pair_name))

            pair_idx += 1

        except Exception as exc:
            log.error("Failed to process pair %s <-> %s: %s", cloudy_path.name, clear_path.name, exc)
            import traceback
            log.error(traceback.format_exc())
            continue

    if not all_patches:
        log.error("No patches extracted! Check your dataset and config.")
        return {"train": 0, "validation": 0, "test": 0}

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_patches)

    n = len(all_patches)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    splits = {
        "train":      all_patches[:n_train],
        "validation": all_patches[n_train:n_train + n_val],
        "test":       all_patches[n_train + n_val:],
    }

    counts: dict[str, int] = {}
    for split, patches in splits.items():
        for cloudy_data, clear_data, mask_data, name in tqdm(
            patches, desc=f"Saving {split}", unit="patch", leave=False
        ):
            np.save(output_dir / split / "cloudy" / f"{name}_cloudy.npy", cloudy_data)
            np.save(output_dir / split / "clear"  / f"{name}_clear.npy",  clear_data)
            np.save(output_dir / split / "masks"  / f"{name}_mask.npy",   mask_data)
        counts[split] = len(patches)
        log.info("Saved %d patches to %s/", len(patches), split)

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare cloud reconstruction dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    source_dir = Path(cfg["dataset"]["source_dir"])
    output_dir = Path(cfg["dataset"]["output_dir"])

    if not source_dir.exists():
        log.error("Source directory not found: %s", source_dir)
        return 1

    log.info("Source: %s", source_dir.resolve())
    log.info("Output: %s", output_dir.resolve())

    # Step 1: Discover scenes
    scenes = discover_scenes(source_dir)
    if not scenes:
        log.error("No scenes found in %s", source_dir)
        return 1

    # Step 2: Group by tile
    tiles = group_by_tile(scenes)
    log.info("Tiles found: %s", list(tiles.keys()))
    for tile_id, by_cls in tiles.items():
        for cls, paths in by_cls.items():
            log.info("  %s / %s: %d scenes", tile_id, cls, len(paths))

    # Step 3: Create pairs
    pairs = create_pairs(
        tiles, 
        source_dir=source_dir,
        output_dir=output_dir,
        include_medium=cfg["dataset"]["include_medium"]
    )
    if not pairs:
        log.error("No cloudy-clear pairs could be formed. Ensure you have matching tile IDs.")
        return 1
    log.info("Created %d training pairs.", len(pairs))

    # Step 4: Extract patches
    counts = extract_and_save_patches(pairs, cfg, output_dir)

    # Summary
    total = sum(counts.values())
    log.info("=" * 50)
    log.info("DATASET PREPARATION COMPLETE")
    log.info("=" * 50)
    log.info("Total patches: %d", total)
    for split, count in counts.items():
        log.info("  %s: %d patches", split, count)
    log.info("Saved to: %s", output_dir.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())
