# -*- coding: utf-8 -*-
"""
sentinel_organizer.py
─────────────────────
Automatically organise Sentinel-2 .SAFE.zip datasets into
  dataset/clear   (cloud cover ≤ clear_threshold)
  dataset/medium  (clear_threshold < cloud cover < cloudy_threshold)
  dataset/cloudy  (cloud cover ≥ cloudy_threshold)

Features
--------
- Reads cloud-cover from MTD_MSIL2A.xml inside each .SAFE.zip
- Generates a 512×512 RGB JPEG preview from the TCI_60m.jp2 band
- Detects and skips exact duplicates (same file size + name)
- Skips corrupted zips gracefully
- Writes a CSV report  (Filename, Date, CloudCover%, Classification)
- Configurable thresholds  (defaults: clear ≤10 %, cloudy ≥70 %)
- Rich tqdm progress bar
- Works entirely in-memory for metadata; moves/copies zips into dataset/

Usage
-----
  python sentinel_organizer.py                          # defaults
  python sentinel_organizer.py --clear 15 --cloudy 60  # custom thresholds
  python sentinel_organizer.py --input /path/to/folder --copy
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import re
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Optional rasterio import (used for GeoTIFF; not needed for .SAFE.zip) ──────
try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIRS: list[str] = [
    r"C:\Users\aksha\Downloads\Cloud",
    r"C:\Users\aksha\Downloads\Cloud free",
]
DEFAULT_OUTPUT_DIR: str = r"C:\Users\aksha\Documents\AntiGravity\earth-satellites\dataset"
DEFAULT_CLEAR_THRESHOLD: float = 10.0
DEFAULT_CLOUDY_THRESHOLD: float = 70.0

PREVIEW_SIZE: tuple[int, int] = (512, 512)
PREVIEW_QUALITY: int = 90

# Ensure stdout handles UTF-8 on Windows terminals
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneInfo:
    source_path: Path
    filename: str
    acq_date: str = "unknown"
    cloud_cover: Optional[float] = None
    classification: str = "unknown"
    preview_path: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-2 filename parser
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_PATTERN = re.compile(r"S2[A-C]_MSIL2A_(?P<datetime>\d{8}T\d{6})_", re.IGNORECASE)


def parse_acquisition_date(name: str) -> str:
    m = _SAFE_PATTERN.search(name)
    if m:
        dt_str = m.group("datetime")
        return f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Cloud-cover extraction from .SAFE.zip
# ─────────────────────────────────────────────────────────────────────────────

def extract_cloud_cover_from_safe_zip(zip_path: Path) -> tuple[float, str]:
    """Returns (cloud_cover_percent, acq_date) from MTD_MSIL2A.xml in the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        mtd_entries = [
            n for n in names
            if "MTD_MSIL2A.xml" in n and n.count("/") <= 2
        ]
        if not mtd_entries:
            raise ValueError("MTD_MSIL2A.xml not found in zip")
        xml_bytes = zf.read(mtd_entries[0])

    tree = ET.fromstring(xml_bytes)
    cloud_cover: Optional[float] = None
    acq_date: str = "unknown"

    for elem in tree.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "Cloud_Coverage_Assessment" and elem.text:
            try:
                cloud_cover = float(elem.text)
            except ValueError:
                pass
        if tag in ("DATATAKE_SENSING_START", "GENERATION_TIME") and elem.text:
            if acq_date == "unknown":
                raw = elem.text.strip()
                acq_date = raw[:10]

    if cloud_cover is None:
        raise ValueError("Cloud_Coverage_Assessment tag not found in MTD_MSIL2A.xml")

    return cloud_cover, acq_date


# ─────────────────────────────────────────────────────────────────────────────
# RGB preview generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_preview_from_safe_zip(
    zip_path: Path,
    out_dir: Path,
    scene_name: str,
    size: tuple[int, int] = PREVIEW_SIZE,
    quality: int = PREVIEW_QUALITY,
) -> Optional[Path]:
    """Extract TCI_60m.jp2 from zip and save as a resized JPEG preview."""
    preview_name = f"{scene_name}_preview.jpg"
    preview_path = out_dir / preview_name
    if preview_path.exists():
        return preview_path

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            tci_candidates = sorted(
                [n for n in names if "TCI" in n and n.endswith(".jp2")],
                key=lambda n: ("60m" not in n, "20m" not in n),
            )
            if not tci_candidates:
                log.warning("No TCI .jp2 found in %s — skipping preview", zip_path.name)
                return None
            raw_bytes = zf.read(tci_candidates[0])

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        out_dir.mkdir(parents=True, exist_ok=True)
        img.save(preview_path, "JPEG", quality=quality, optimize=True)
        return preview_path

    except Exception as exc:
        log.warning("Preview generation failed for %s: %s", zip_path.name, exc)
        return None


def generate_preview_from_geotiff(
    tif_path: Path,
    out_dir: Path,
    scene_name: str,
    size: tuple[int, int] = PREVIEW_SIZE,
    quality: int = PREVIEW_QUALITY,
) -> Optional[Path]:
    """Build an RGB JPEG preview from a multi-band GeoTIFF using rasterio."""
    if not HAS_RASTERIO:
        return None

    preview_name = f"{scene_name}_preview.jpg"
    preview_path = out_dir / preview_name
    if preview_path.exists():
        return preview_path

    try:
        with rasterio.open(tif_path) as src:
            scale = max(1, src.width // size[0])
            out_w, out_h = src.width // scale, src.height // scale

            def _read(band: int) -> np.ndarray:
                return src.read(
                    min(band, src.count),
                    out_shape=(out_h, out_w),
                    resampling=Resampling.lanczos,
                ).astype(np.float32)

            r, g, b = _read(1), _read(2), _read(3)

        def _norm(arr: np.ndarray) -> np.ndarray:
            valid = arr[arr > 0]
            p2, p98 = (np.percentile(valid, (2, 98)) if valid.size else (0, 1))
            return np.clip((arr - p2) / max(p98 - p2, 1e-6) * 255, 0, 255).astype(np.uint8)

        rgb = np.dstack([_norm(r), _norm(g), _norm(b)])
        img = Image.fromarray(rgb, "RGB").resize(size, Image.LANCZOS)
        out_dir.mkdir(parents=True, exist_ok=True)
        img.save(preview_path, "JPEG", quality=quality, optimize=True)
        return preview_path

    except Exception as exc:
        log.warning("Preview (GeoTIFF) failed for %s: %s", tif_path.name, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cloud cover from GeoTIFF sidecar XML / rasterio tags
# ─────────────────────────────────────────────────────────────────────────────

def extract_cloud_cover_from_geotiff(tif_path: Path) -> tuple[float, str]:
    for suf in (".xml", "_MTD.xml", "_metadata.xml"):
        meta_path = tif_path.with_suffix(suf)
        if meta_path.exists():
            try:
                tree = ET.parse(meta_path)
                for elem in tree.iter():
                    tag = elem.tag.split("}")[-1]
                    if tag == "Cloud_Coverage_Assessment" and elem.text:
                        return float(elem.text), parse_acquisition_date(tif_path.stem)
            except Exception:
                pass

    if HAS_RASTERIO:
        with rasterio.open(tif_path) as src:
            tags = src.tags()
            for key in ("CLOUD_COVER", "Cloud_Coverage_Assessment", "cloud_cover"):
                if key in tags:
                    return float(tags[key]), parse_acquisition_date(tif_path.stem)

    raise ValueError(f"Cannot determine cloud cover for GeoTIFF: {tif_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify(cloud_cover: float, clear_thresh: float, cloudy_thresh: float) -> str:
    if cloud_cover <= clear_thresh:
        return "clear"
    if cloud_cover >= cloudy_thresh:
        return "cloudy"
    return "medium"


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate detection (filename stem + file size — fast, no full-file hash)
# ─────────────────────────────────────────────────────────────────────────────

def file_signature(path: Path) -> str:
    stem = re.sub(r"\.SAFE(\.zip)?$", "", path.name, flags=re.IGNORECASE)
    return f"{stem}::{path.stat().st_size}"


# ─────────────────────────────────────────────────────────────────────────────
# Scene discovery across multiple input directories
# ─────────────────────────────────────────────────────────────────────────────

def discover_scenes(input_dirs: list[Path]) -> list[Path]:
    """Return de-duplicated list of .SAFE.zip and GeoTIFF files."""
    seen_sigs: set[str] = set()
    scenes: list[Path] = []

    for d in input_dirs:
        if not d.exists():
            log.warning("Input directory does not exist: %s", d)
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            is_safe_zip = f.name.lower().endswith(".safe.zip") or (
                f.name.lower().endswith(".zip") and "MSIL2A" in f.name
            )
            is_tiff = f.suffix.lower() in (".tif", ".tiff")
            if not (is_safe_zip or is_tiff):
                continue

            sig = file_signature(f)
            if sig in seen_sigs:
                log.info("[SKIP-DUP] Duplicate skipped: %s", f.name)
                continue
            seen_sigs.add(sig)
            scenes.append(f)

    return scenes


# ─────────────────────────────────────────────────────────────────────────────
# Core scene processing
# ─────────────────────────────────────────────────────────────────────────────

def process_scene(
    scene_path: Path,
    output_dir: Path,
    clear_thresh: float,
    cloudy_thresh: float,
    copy_mode: bool,
    previews_dir: Path,
    skip_preview: bool,
) -> SceneInfo:
    info = SceneInfo(source_path=scene_path, filename=scene_path.name)
    info.acq_date = parse_acquisition_date(scene_path.stem)
    is_zip = scene_path.name.lower().endswith(".zip")
    scene_stem = re.sub(r"\.SAFE(\.zip)?$", "", scene_path.name, flags=re.IGNORECASE)

    # 1 — Validate zip integrity
    if is_zip:
        try:
            with zipfile.ZipFile(scene_path, "r") as zf:
                bad = zf.testzip()
            if bad:
                info.error = f"Corrupted zip: first bad file = {bad}"
                info.skipped, info.skip_reason = True, "corrupted"
                return info
        except zipfile.BadZipFile as exc:
            info.error = str(exc)
            info.skipped, info.skip_reason = True, "bad zip"
            return info

    # 2 — Extract cloud cover from metadata
    try:
        if is_zip:
            info.cloud_cover, date_from_xml = extract_cloud_cover_from_safe_zip(scene_path)
            if info.acq_date == "unknown":
                info.acq_date = date_from_xml
        else:
            info.cloud_cover, _ = extract_cloud_cover_from_geotiff(scene_path)
    except Exception as exc:
        info.error = f"Metadata error: {exc}"
        info.skipped, info.skip_reason = True, "metadata error"
        return info

    # 3 — Classify
    info.classification = classify(info.cloud_cover, clear_thresh, cloudy_thresh)

    # 4 — Ensure destination dirs exist
    dest_class_dir = output_dir / info.classification
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_class_dir / scene_path.name

    # 5 — Generate RGB preview
    if not skip_preview:
        preview_out = previews_dir / info.classification
        preview_out.mkdir(parents=True, exist_ok=True)
        if is_zip:
            info.preview_path = generate_preview_from_safe_zip(scene_path, preview_out, scene_stem)
        else:
            info.preview_path = generate_preview_from_geotiff(scene_path, preview_out, scene_stem)

    # 6 — Move or copy
    if not dest_file.exists():
        if copy_mode:
            shutil.copy2(scene_path, dest_file)
        else:
            shutil.move(str(scene_path), dest_file)

    return info


# ─────────────────────────────────────────────────────────────────────────────
# CSV report
# ─────────────────────────────────────────────────────────────────────────────

def write_csv_report(results: list[SceneInfo], output_dir: Path) -> Path:
    report_path = output_dir / "classification_report.csv"
    fieldnames = [
        "Filename", "Acquisition Date", "Cloud Cover %",
        "Classification", "Preview", "Status", "Notes",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Filename": r.filename,
                "Acquisition Date": r.acq_date,
                "Cloud Cover %": f"{r.cloud_cover:.2f}" if r.cloud_cover is not None else "N/A",
                "Classification": r.classification,
                "Preview": str(r.preview_path) if r.preview_path else "N/A",
                "Status": "skipped" if r.skipped else "ok",
                "Notes": r.skip_reason or r.error or "",
            })
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[SceneInfo], clear_thresh: float, cloudy_thresh: float, elapsed: float) -> None:
    ok = [r for r in results if not r.skipped]
    skip = [r for r in results if r.skipped]
    counts = {k: sum(1 for r in ok if r.classification == k) for k in ("clear", "medium", "cloudy")}

    print("\n" + "═" * 62)
    print("  SENTINEL-2 ORGANIZER — SUMMARY")
    print("═" * 62)
    print(f"  Total scenes found    : {len(results)}")
    print(f"  Processed OK          : {len(ok)}")
    print(f"  Skipped / errored     : {len(skip)}")
    print(f"  [SUN] clear  (<= {clear_thresh}%)  : {counts['clear']}")
    print(f"  [MID] medium              : {counts['medium']}")
    print(f"  [CLD] cloudy (>= {cloudy_thresh}%) : {counts['cloudy']}")
    print(f"  Elapsed               : {elapsed:.1f}s")
    print("═" * 62)
    if skip:
        print("\n  Skipped files:")
        for r in skip:
            print(f"    • {r.filename}  [{r.skip_reason}]")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sentinel-2 dataset organiser — classifies scenes by cloud cover.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", nargs="+", default=DEFAULT_INPUT_DIRS, metavar="DIR",
                   help="Input directories containing .SAFE.zip or GeoTIFF files.")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
                   help="Root output directory (clear/medium/cloudy created inside).")
    p.add_argument("--clear", type=float, default=DEFAULT_CLEAR_THRESHOLD, metavar="PCT",
                   help="Cloud cover ≤ this  → 'clear'.")
    p.add_argument("--cloudy", type=float, default=DEFAULT_CLOUDY_THRESHOLD, metavar="PCT",
                   help="Cloud cover ≥ this  → 'cloudy'.")
    p.add_argument("--copy", action="store_true", default=False,
                   help="Copy files (keep originals). Default: move.")
    p.add_argument("--no-preview", action="store_true", default=False,
                   help="Skip RGB preview generation.")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import time

    args = build_parser().parse_args(argv)

    if args.clear >= args.cloudy:
        log.error("--clear (%.1f) must be less than --cloudy (%.1f)", args.clear, args.cloudy)
        return 2

    input_dirs = [Path(d) for d in args.input]
    output_dir = Path(args.output)
    previews_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Scanning: %s", [str(d) for d in input_dirs])
    scenes = discover_scenes(input_dirs)

    if not scenes:
        log.warning("No Sentinel-2 scenes found. Exiting.")
        return 1

    log.info("Found %d unique scenes. clear≤%.0f%% | cloudy≥%.0f%%",
             len(scenes), args.clear, args.cloudy)

    results: list[SceneInfo] = []
    t0 = time.time()

    with tqdm(scenes, desc="Processing", unit="scene", dynamic_ncols=True, colour="cyan") as pbar:
        for scene_path in pbar:
            pbar.set_postfix_str(scene_path.name[:52])
            try:
                info = process_scene(
                    scene_path=scene_path,
                    output_dir=output_dir,
                    clear_thresh=args.clear,
                    cloudy_thresh=args.cloudy,
                    copy_mode=args.copy,
                    previews_dir=previews_dir,
                    skip_preview=args.no_preview,
                )
            except Exception:
                info = SceneInfo(
                    source_path=scene_path,
                    filename=scene_path.name,
                    error=traceback.format_exc(),
                    skipped=True,
                    skip_reason="unexpected error",
                )
                log.error("Unexpected error processing %s", scene_path.name)

            results.append(info)

            icon = {"clear": "[SUN]", "medium": "[MID]", "cloudy": "[CLD]"}.get(info.classification, "[???]")
            if info.skipped:
                pbar.write(f"  [SKIP]  {info.filename}  [{info.skip_reason}]")
            else:
                pbar.write(
                    f"  {icon} {info.classification.upper():<7} "
                    f"{info.cloud_cover:6.2f}%  {info.filename}"
                )

    elapsed = time.time() - t0
    report_path = write_csv_report(results, output_dir)
    log.info("Report: %s", report_path)
    print_summary(results, args.clear, args.cloudy, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
