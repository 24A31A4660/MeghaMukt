"""evaluation/metrics.py — PSNR, SSIM, RMSE, MAE, SAM, LPIPS on cloud-only and full image."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


@dataclass
class MetricsResult:
    """Holds per-scene evaluation metrics."""
    psnr_full:      float
    ssim_full:      float
    rmse_full:      float
    mae_full:       float
    sam_full:       float
    lpips_full:     float
    psnr_cloud:     float   # computed only on cloud-masked pixels
    ssim_cloud:     float
    rmse_cloud:     float
    cloud_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "psnr_full":      round(self.psnr_full, 4),
            "ssim_full":      round(self.ssim_full, 4),
            "rmse_full":      round(self.rmse_full, 4),
            "mae_full":       round(self.mae_full, 4),
            "sam_full":       round(self.sam_full, 4),
            "lpips_full":     round(self.lpips_full, 4),
            "psnr_cloud":     round(self.psnr_cloud, 4),
            "ssim_cloud":     round(self.ssim_cloud, 4),
            "rmse_cloud":     round(self.rmse_cloud, 4),
            "cloud_fraction": round(self.cloud_fraction, 4),
        }

    def summary(self) -> str:
        return (
            f"PSNR(full)={self.psnr_full:.2f}dB  SSIM(full)={self.ssim_full:.4f}  "
            f"RMSE(full)={self.rmse_full:.4f}  MAE(full)={self.mae_full:.4f}  "
            f"SAM(full)={self.sam_full:.2f}°  LPIPS(full)={self.lpips_full:.4f}  |  "
            f"PSNR(cloud)={self.psnr_cloud:.2f}dB  SSIM(cloud)={self.ssim_cloud:.4f}  "
            f"RMSE(cloud)={self.rmse_cloud:.4f}"
        )


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR in dB. Higher = better."""
    return float(_psnr(target, pred, data_range=data_range))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """
    SSIM averaged over all channels.
    Expects [H, W] or [H, W, C].
    """
    if pred.ndim == 3:
        # Average SSIM over channels
        scores = [
            _ssim(target[..., c], pred[..., c], data_range=data_range)
            for c in range(pred.shape[-1])
        ]
        return float(np.mean(scores))
    return float(_ssim(target, pred, data_range=data_range))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Square Error. Lower = better."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error. Lower = better."""
    return float(np.mean(np.abs(pred - target)))


def compute_sam(pred: np.ndarray, target: np.ndarray) -> float:
    """Spectral Angle Mapper (in degrees). Lower = better.

    Computes the average spectral angle between predicted and target spectra
    across all pixels. Expects [H, W, C] arrays (C = spectral bands).

    For single-channel or 2D arrays, returns 0.0 (not meaningful).
    """
    if pred.ndim < 3 or pred.shape[-1] < 2:
        return 0.0

    # Flatten spatial dims: [N, C]
    p = pred.reshape(-1, pred.shape[-1]).astype(np.float64)
    t = target.reshape(-1, target.shape[-1]).astype(np.float64)

    # Spectral angle per pixel
    dot = np.sum(p * t, axis=-1)
    norm_p = np.linalg.norm(p, axis=-1)
    norm_t = np.linalg.norm(t, axis=-1)

    # Avoid division by zero
    denom = norm_p * norm_t
    valid = denom > 1e-10
    cos_angle = np.zeros_like(dot)
    cos_angle[valid] = dot[valid] / denom[valid]

    # Clamp to [-1, 1] for arccos stability
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(np.nanmean(angle_deg[valid])) if valid.any() else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS — Learned Perceptual Image Patch Similarity
# ─────────────────────────────────────────────────────────────────────────────

# Lazy-loaded singleton to avoid importing torch at module level
_lpips_model = None
_lpips_available: Optional[bool] = None


def _get_lpips_model():
    """Lazy-load LPIPS model (VGG-based, lightweight)."""
    global _lpips_model, _lpips_available
    if _lpips_available is not None:
        return _lpips_model

    try:
        import lpips
        _lpips_model = lpips.LPIPS(net="vgg", verbose=False)
        _lpips_model.eval()
        # Move to CPU for metric computation (not part of training loop)
        _lpips_available = True
    except ImportError:
        _lpips_available = False
        _lpips_model = None
        # Fallback: try torchmetrics
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            _lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
            _lpips_model.eval()
            _lpips_available = True
        except ImportError:
            _lpips_available = False
            _lpips_model = None
    except Exception:
        _lpips_available = False
        _lpips_model = None

    return _lpips_model


def compute_lpips(pred: np.ndarray, target: np.ndarray) -> float:
    """Learned Perceptual Image Patch Similarity. Lower = better.

    Expects [H, W, C] or [C, H, W] arrays in [0, 1].
    Uses the RGB subset (first 3 channels or R,G,B from Sentinel-2 band order).

    Falls back to 0.0 if lpips/torchmetrics is not installed.
    """
    model = _get_lpips_model()
    if model is None:
        return 0.0

    try:
        import torch

        # Ensure [H, W, C]
        if pred.ndim == 3 and pred.shape[0] < pred.shape[-1]:
            pred = np.transpose(pred, (1, 2, 0))
            target = np.transpose(target, (1, 2, 0))

        # Extract RGB (Sentinel-2 band order: B02=0, B03=1, B04=2 → R=ch2, G=ch1, B=ch0)
        if pred.shape[-1] >= 3:
            pred_rgb = np.stack([pred[..., 2], pred[..., 1], pred[..., 0]], axis=-1)
            target_rgb = np.stack([target[..., 2], target[..., 1], target[..., 0]], axis=-1)
        else:
            # Grayscale: repeat to 3 channels
            pred_rgb = np.repeat(pred[..., :1], 3, axis=-1)
            target_rgb = np.repeat(target[..., :1], 3, axis=-1)

        # LPIPS expects [B, 3, H, W] in [-1, 1]
        pred_t = torch.from_numpy(pred_rgb).float().permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        target_t = torch.from_numpy(target_rgb).float().permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

        with torch.no_grad():
            score = model(pred_t, target_t)

        return float(score.item())
    except Exception:
        return 0.0


def compute_metrics(
    pred:       np.ndarray,    # [C, H, W] or [H, W, C] float32 in [0, 1]
    target:     np.ndarray,    # same shape
    cloud_mask: np.ndarray,    # [H, W] binary uint8
    data_range: float = 1.0,
) -> MetricsResult:
    """
    Compute all metrics on both full image and cloud-only region.

    Parameters
    ----------
    pred, target : Arrays in [0, 1].
    cloud_mask   : Binary mask, 1 = cloud pixel.
    """
    # Ensure [H, W, C] for skimage
    if pred.ndim == 3 and pred.shape[0] < pred.shape[-1]:
        pred_hwc   = np.transpose(pred, (1, 2, 0))
        target_hwc = np.transpose(target, (1, 2, 0))
    else:
        pred_hwc   = pred
        target_hwc = target

    cloud_fraction = float(np.mean(cloud_mask > 0))

    # Full image metrics
    psnr_full  = compute_psnr(pred_hwc, target_hwc, data_range)
    ssim_full  = compute_ssim(pred_hwc, target_hwc, data_range)
    rmse_full  = compute_rmse(pred_hwc, target_hwc)
    mae_full   = compute_mae(pred_hwc, target_hwc)
    sam_full   = compute_sam(pred_hwc, target_hwc)
    lpips_full = compute_lpips(pred_hwc, target_hwc)

    # Cloud-region-only metrics
    mask = (cloud_mask > 0)
    if mask.any():
        if pred_hwc.ndim == 3:
            pred_cloud   = pred_hwc[mask]    # [N_cloud, C]
            target_cloud = target_hwc[mask]  # [N_cloud, C]
        else:
            pred_cloud   = pred_hwc[mask]
            target_cloud = target_hwc[mask]

        rmse_cloud = float(np.sqrt(np.mean((pred_cloud - target_cloud) ** 2)))
        # PSNR on cloud pixels only
        mse_cloud = np.mean((pred_cloud - target_cloud) ** 2)
        psnr_cloud = float(10 * np.log10(data_range ** 2 / max(mse_cloud, 1e-10)))
        # SSIM: crop bounding box of cloud region for spatial metric
        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        pred_crop   = pred_hwc[r0:r1, c0:c1]
        target_crop = target_hwc[r0:r1, c0:c1]
        ssim_cloud = compute_ssim(pred_crop, target_crop, data_range)
    else:
        psnr_cloud = psnr_full
        ssim_cloud = ssim_full
        rmse_cloud = rmse_full

    return MetricsResult(
        psnr_full=psnr_full,
        ssim_full=ssim_full,
        rmse_full=rmse_full,
        mae_full=mae_full,
        sam_full=sam_full,
        lpips_full=lpips_full,
        psnr_cloud=psnr_cloud,
        ssim_cloud=ssim_cloud,
        rmse_cloud=rmse_cloud,
        cloud_fraction=cloud_fraction,
    )
