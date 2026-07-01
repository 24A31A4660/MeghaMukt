"""evaluation/metrics.py — PSNR, SSIM, RMSE on cloud-only and full image."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


@dataclass
class MetricsResult:
    """Holds per-scene evaluation metrics."""
    psnr_full:      float
    ssim_full:      float
    rmse_full:      float
    psnr_cloud:     float   # computed only on cloud-masked pixels
    ssim_cloud:     float
    rmse_cloud:     float
    cloud_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "psnr_full":      round(self.psnr_full, 4),
            "ssim_full":      round(self.ssim_full, 4),
            "rmse_full":      round(self.rmse_full, 4),
            "psnr_cloud":     round(self.psnr_cloud, 4),
            "ssim_cloud":     round(self.ssim_cloud, 4),
            "rmse_cloud":     round(self.rmse_cloud, 4),
            "cloud_fraction": round(self.cloud_fraction, 4),
        }

    def summary(self) -> str:
        return (
            f"PSNR(full)={self.psnr_full:.2f}dB  SSIM(full)={self.ssim_full:.4f}  "
            f"RMSE(full)={self.rmse_full:.4f}  |  "
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
    psnr_full = compute_psnr(pred_hwc, target_hwc, data_range)
    ssim_full = compute_ssim(pred_hwc, target_hwc, data_range)
    rmse_full = compute_rmse(pred_hwc, target_hwc)

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
        psnr_cloud=psnr_cloud,
        ssim_cloud=ssim_cloud,
        rmse_cloud=rmse_cloud,
        cloud_fraction=cloud_fraction,
    )
