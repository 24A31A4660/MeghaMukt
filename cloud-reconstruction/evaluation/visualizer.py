"""evaluation/visualizer.py — 7-panel comparison images and output saving."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
# Colour Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a [C, H, W] or [H, W] float32 array to [H, W, 3] uint8 RGB.
    For multi-band: picks bands 2,1,0 (B04-R, B03-G, B02-B).
    For single-channel: repeats to greyscale.
    """
    if arr.ndim == 2:
        p2, p98 = np.percentile(arr, (2, 98))
        scaled = np.clip((arr - p2) / max(p98 - p2, 1e-6), 0, 1)
        rgb = np.stack([scaled] * 3, axis=-1)
    elif arr.shape[0] >= 3:
        if arr.shape[0] == 3:
            r, g, b = arr[0], arr[1], arr[2]
        else:
            r, g, b = arr[2], arr[1], arr[0]   # Sentinel-2: B04, B03, B02 → R,G,B
        rgb = np.stack([r, g, b], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0, 1)
    else:
        rgb = np.transpose(arr[:3], (1, 2, 0))
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _to_mask_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert binary [H, W] mask to coloured overlay [H, W, 3] uint8."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask > 0] = [255, 80, 80]    # Red for cloud
    rgb[mask == 0] = [40, 40, 40]    # Dark grey for clear
    return rgb


def _to_heatmap(arr: np.ndarray) -> np.ndarray:
    """Convert a [H, W] float32 [0,1] map to a blue-red heatmap [H, W, 3] uint8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    cmap = cm.get_cmap("RdYlBu_r")
    rgba = cmap(np.clip(arr, 0, 1))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    """Load a system font, falling back to default if unavailable."""
    for font_path in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, AttributeError):
            continue
    return ImageFont.load_default()


def _add_label(img: Image.Image, text: str, color: str = "white",
               font_size: int = 14) -> Image.Image:
    """Add a semi-transparent label bar at the top of an image."""
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    draw.rectangle([(0, 0), (img.width, 22)], fill=(0, 0, 0, 180))
    draw.text((4, 3), text, fill=color, font=font)
    return img


def _metrics_panel(metrics: dict, width: int, height: int) -> np.ndarray:
    """Render a metrics summary panel as a PIL image → [H, W, 3] uint8."""
    img = Image.new("RGB", (width, height), (18, 18, 24))
    draw = ImageDraw.Draw(img)
    font_title = _load_font(15)
    font_body  = _load_font(13)

    draw.text((8, 6), "Metrics", fill=(180, 220, 255), font=font_title)

    metric_labels = {
        "psnr_full":  ("PSNR",  "dB",  (120, 230, 120)),
        "ssim_full":  ("SSIM",  "",    (120, 200, 255)),
        "rmse_full":  ("RMSE",  "",    (255, 180, 80)),
        "mae_full":   ("MAE",   "",    (255, 140, 200)),
        "sam_full":   ("SAM",   "°",   (200, 160, 255)),
        "lpips_full": ("LPIPS", "",    (255, 100, 100)),
        "psnr_cloud": ("PSNR↑", "dB (cloud)", (120, 230, 120)),
        "ssim_cloud": ("SSIM↑", "(cloud)",    (120, 200, 255)),
        "rmse_cloud": ("RMSE↓", "(cloud)",    (255, 180, 80)),
        "cloud_fraction": ("Cloud", "%",      (200, 200, 200)),
    }

    y = 28
    for key, (label, unit, color) in metric_labels.items():
        if key in metrics:
            val = metrics[key]
            if key == "cloud_fraction":
                val_str = f"{val * 100:.1f}%"
            elif unit == "dB":
                val_str = f"{val:.2f} {unit}"
            elif unit == "°":
                val_str = f"{val:.2f}{unit}"
            else:
                val_str = f"{val:.4f}"
            line = f"{label}: {val_str}"
            draw.text((8, y), line, fill=color, font=font_body)
            y += 18
            if y > height - 10:
                break

    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# Main Comparison Panel
# ─────────────────────────────────────────────────────────────────────────────

def create_comparison_panel(
    cloudy_input:   np.ndarray,         # [C, H, W]
    cloud_mask:     np.ndarray,         # [H, W]
    prediction:     np.ndarray,         # [C, H, W]
    ground_truth:   np.ndarray | None,  # [C, H, W] or None
    difference_map: np.ndarray | None,  # [H, W] or None
    confidence_map: np.ndarray | None = None,   # [H, W] or None
    metrics:        dict | None = None,
    output_path:    Path | None = None,
    panel_size:     int = 256,
) -> Image.Image:
    """
    Create a 7-panel comparison image:

    | Cloudy | Mask | Prediction | Ground Truth | Difference | Confidence | Metrics |

    If ground_truth is None (inference mode), the Ground Truth panel is replaced
    by the Confidence Map and the Metrics panel is replaced by scene info.

    Returns: PIL.Image — the assembled comparison strip.
    """
    panels: list[tuple[np.ndarray, str]] = [
        (_to_rgb_uint8(cloudy_input), "Cloudy Input"),
        (_to_mask_rgb(cloud_mask),    "Cloud Mask"),
        (_to_rgb_uint8(prediction),   "Prediction"),
    ]

    if ground_truth is not None:
        panels.append((_to_rgb_uint8(ground_truth), "Ground Truth"))

    if difference_map is not None:
        panels.append((_to_heatmap(difference_map), "Difference Map"))
    elif confidence_map is not None:
        panels.append((_to_heatmap(confidence_map), "Confidence Map"))

    if confidence_map is not None and difference_map is not None:
        panels.append((_to_heatmap(confidence_map), "Confidence Map"))

    # Assemble image panels
    pil_panels: list[Image.Image] = []
    for arr, label in panels:
        img = Image.fromarray(arr).resize((panel_size, panel_size), Image.LANCZOS)
        img = img.convert("RGBA")
        img = _add_label(img, label)
        pil_panels.append(img)

    # Add metrics panel if metrics provided
    if metrics:
        met_arr = _metrics_panel(metrics, panel_size, panel_size)
        met_img = Image.fromarray(met_arr).convert("RGBA")
        met_img = _add_label(met_img, "Metrics", color=(180, 220, 255))
        pil_panels.append(met_img)

    gap = 4
    total_w = panel_size * len(pil_panels) + gap * (len(pil_panels) - 1)
    strip = Image.new("RGBA", (total_w, panel_size), (20, 20, 20, 255))

    x = 0
    for panel in pil_panels:
        strip.paste(panel, (x, 0))
        x += panel_size + gap

    strip_rgb = strip.convert("RGB")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        strip_rgb.save(output_path, "JPEG", quality=92, optimize=True)

    return strip_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Full Output Saver
# ─────────────────────────────────────────────────────────────────────────────

def save_all_outputs(
    scene_name:     str,
    output_dir:     Path,
    cloudy_input:   np.ndarray,         # [C, H, W]
    prediction:     np.ndarray,         # [C, H, W]
    cloud_mask:     np.ndarray,         # [H, W]
    confidence_map: np.ndarray,         # [H, W]
    difference_map: np.ndarray | None,  # [H, W]
    ground_truth:   np.ndarray | None,
    metrics:        dict | None = None,
    meta=None,
) -> dict[str, Path]:
    """Save all inference outputs to disk. Returns dict of name → Path."""
    from utils.geotiff import write_geotiff, write_rgb_preview
    import json

    scene_dir = Path(output_dir) / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # 1. Multi-band GeoTIFF
    if meta is not None:
        out_tif = scene_dir / "cloud_free.tif"
        write_geotiff(
            out_tif, prediction, meta,
            band_descriptions=["B02", "B03", "B04", "B08", "B11", "B12"],
        )
        outputs["cloud_free_geotiff"] = out_tif

    # 2. RGB preview JPEG
    rgb_path = scene_dir / "cloud_free_rgb.jpg"
    write_rgb_preview(rgb_path, prediction)
    outputs["cloud_free_rgb"] = rgb_path

    # 3. Cloud mask
    mask_path = scene_dir / "cloud_mask.png"
    Image.fromarray(_to_mask_rgb(cloud_mask)).save(mask_path)
    outputs["cloud_mask"] = mask_path

    # 4. Confidence map
    conf_path = scene_dir / "confidence_map.png"
    Image.fromarray(_to_heatmap(confidence_map)).save(conf_path)
    outputs["confidence_map"] = conf_path

    # 5. Difference map
    if difference_map is not None:
        diff_path = scene_dir / "difference_map.png"
        Image.fromarray(_to_heatmap(difference_map)).save(diff_path)
        outputs["difference_map"] = diff_path

    # 6. Comparison panel (7-panel with metrics)
    panel_path = scene_dir / "comparison.jpg"
    create_comparison_panel(
        cloudy_input=cloudy_input,
        cloud_mask=cloud_mask,
        prediction=prediction,
        ground_truth=ground_truth,
        difference_map=difference_map,
        confidence_map=confidence_map,
        metrics=metrics,
        output_path=panel_path,
    )
    outputs["comparison"] = panel_path

    # 7. Metrics JSON
    if metrics is not None:
        metrics_path = scene_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        outputs["metrics"] = metrics_path

    # 8. Processing log
    log_path = scene_dir / "processing_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Scene: {scene_name}\n")
        f.write(f"Cloud fraction: {float(np.mean(cloud_mask > 0)):.2%}\n")
        if metrics:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        f.write(f"Outputs: {list(outputs.keys())}\n")
    outputs["log"] = log_path

    return outputs
