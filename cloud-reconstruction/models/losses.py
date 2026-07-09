"""models/losses.py — Combined reconstruction loss: L1 + SSIM + Perceptual + Charbonnier.

Loss weights (user specification):
    Total = 0.5 × L1  +  0.3 × SSIM  +  0.1 × Perceptual  +  0.1 × Charbonnier

L1 is computed on both cloud and clear regions (identity enforcement).
SSIM is computed globally (sliding-window metric).
Perceptual is computed on the VGG16 RGB feature space, masked to cloud regions.
Charbonnier is a smooth L1 variant that preserves edges better than standard L1.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SSIM Loss
# ─────────────────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """Structural Similarity loss using pytorch_msssim."""

    def __init__(self, data_range: float = 1.0, channel: int = 6) -> None:
        super().__init__()
        try:
            from pytorch_msssim import ssim
            self._ssim_fn = ssim
            self._available = True
        except ImportError:
            self._available = False

        self.data_range = data_range
        self.channel = channel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return F.l1_loss(pred, target, reduction='mean')  # fallback
        ssim_val = self._ssim_fn(
            pred, target,
            data_range=self.data_range,
            size_average=True,
        )
        return 1.0 - ssim_val


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Loss (VGG16)
# ─────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG16 feature-matching loss computed on the RGB subset (bands 2,1,0 = R,G,B).
    Only uses early conv layers (relu1_2, relu2_2) for texture/structure.
    Returns a scalar mean loss (not per-pixel).
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            # Use first 9 layers: up to relu2_2
            self.features = nn.Sequential(*list(vgg.features.children())[:9])
            for p in self.features.parameters():
                p.requires_grad = False
            self._available = True
        except Exception:
            self._available = False
            self.features = None

        # ImageNet normalisation (applied to RGB input)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _extract_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Extract B04(R), B03(G), B02(B) → [B, 3, H, W] normalised for VGG."""
        # Sentinel-2 band order: B02=ch0, B03=ch1, B04=ch2
        rgb = torch.stack([x[:, 2], x[:, 1], x[:, 0]], dim=1)  # R,G,B
        return (rgb - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute perceptual loss, optionally masked to cloud regions.

        Args:
            pred:   [B, C, H, W] predicted image.
            target: [B, C, H, W] ground truth image.
            mask:   [B, 1, H, W] cloud mask (1=cloud). If None, full image loss.

        Returns:
            Scalar perceptual loss.
        """
        if not self._available or self.features is None:
            return torch.tensor(0.0, device=pred.device)

        pred_rgb   = self._extract_rgb(pred)
        target_rgb = self._extract_rgb(target)

        pred_feat   = self.features(pred_rgb)
        target_feat = self.features(target_rgb)

        # Compute per-pixel L1 difference in feature space
        feat_diff = torch.abs(pred_feat - target_feat)  # [B, C_feat, H_feat, W_feat]

        if mask is not None:
            # Downsample mask to feature map resolution
            mask_down = F.adaptive_avg_pool2d(mask.float(), feat_diff.shape[2:])
            cloud_area = torch.clamp(mask_down.sum(), min=1.0)
            # Weight loss by cloud mask
            perc_loss = (feat_diff * mask_down).sum() / (cloud_area * feat_diff.shape[1])
        else:
            perc_loss = feat_diff.mean()

        return perc_loss


# ─────────────────────────────────────────────────────────────────────────────
# Charbonnier Loss
# ─────────────────────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1 variant).

    L = sqrt((pred - target)^2 + epsilon^2)

    Behaves like L2 for small errors and L1 for large errors.
    Better than standard L1 at preserving fine details and edges
    because the gradient is smooth near zero (no discontinuity).

    Commonly used in image restoration (SRResNet, EDSR, etc.).
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon_sq = epsilon ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Charbonnier loss, optionally masked to cloud regions.

        Args:
            pred:   [B, C, H, W] predicted image.
            target: [B, C, H, W] ground truth image.
            mask:   [B, 1, H, W] cloud mask (1=cloud). If None, full image loss.

        Returns:
            Scalar Charbonnier loss.
        """
        diff_sq = (pred - target) ** 2
        loss_map = torch.sqrt(diff_sq + self.epsilon_sq)  # [B, C, H, W]

        if mask is not None:
            cloud_area = torch.clamp(mask.sum(), min=1.0)
            loss = (loss_map * mask).sum() / (cloud_area * loss_map.shape[1])
        else:
            loss = loss_map.mean()

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────

class SpectralConsistencyLoss(nn.Module):
    """Spectral consistency loss for Sentinel-2 imagery.

    Sentinel-2 band layout in our tensor:
        ch0 = B02  (Blue)
        ch1 = B03  (Green)
        ch2 = B04  (Red)
        ch3 = B08  (NIR)
        ch4 = B11  (SWIR-1)
        ch5 = B12  (SWIR-2)

    Computes MSE between predicted and target spectral indices:
        1. NDVI  = (NIR - Red) / (NIR + Red + eps)       range [-1, 1]
        2. SR    = NIR / (Red + eps)  (Simple Ratio)      clamped [0, 10]
        3. GSWIR = Green / (SWIR-1 + eps)                 clamped [0, 10]

    All ratios are cloud-masked (loss applies only to reconstructed pixels).
    A small weight (0.05) is enough to steer spectral realism without
    overpowering the pixel-level and structural objectives.

    Why this matters for remote sensing:
        Cloud removal models can produce visually plausible but spectrally
        inconsistent outputs — e.g. vegetation with suppressed NIR, or
        urban surfaces with wrong SWIR signatures.  This loss penalises
        such errors directly in the spectral domain.
    """

    # Band indices (must match config.yaml bands.optical order)
    _RED   = 2   # B04
    _NIR   = 3   # B08
    _GREEN = 1   # B03
    _SWIR1 = 4   # B11

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _ndvi(self, x: torch.Tensor) -> torch.Tensor:
        """NDVI = (NIR - Red) / (NIR + Red).  Output in [-1, 1]."""
        nir = x[:, self._NIR:self._NIR + 1]
        red = x[:, self._RED:self._RED + 1]
        return (nir - red) / (nir + red + self.eps)

    def _simple_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """SR = NIR / Red.  Clamped to [0, 10] to avoid explosion."""
        nir = x[:, self._NIR:self._NIR + 1]
        red = x[:, self._RED:self._RED + 1]
        return torch.clamp(nir / (red + self.eps), 0.0, 10.0)

    def _green_swir(self, x: torch.Tensor) -> torch.Tensor:
        """Green / SWIR-1 ratio.  Clamped to [0, 10]."""
        green = x[:, self._GREEN:self._GREEN + 1]
        swir1 = x[:, self._SWIR1:self._SWIR1 + 1]
        return torch.clamp(green / (swir1 + self.eps), 0.0, 10.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred:   torch.Tensor,                   # [B, 6, H, W]
        target: torch.Tensor,                   # [B, 6, H, W]
        mask:   Optional[torch.Tensor] = None,  # [B, 1, H, W] or [B, H, W]
    ) -> torch.Tensor:
        """Compute spectral consistency MSE.

        Args:
            pred, target : [B, 6, H, W] float tensors in [0, 1].
            mask         : [B, 1, H, W] cloud mask (1=cloud). If None, full image.

        Returns:
            Scalar spectral loss.
        """
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]

        # Stack the 3 spectral indices: [B, 3, H, W]
        indices_pred = torch.cat([
            self._ndvi(pred),
            self._simple_ratio(pred),
            self._green_swir(pred),
        ], dim=1)

        indices_target = torch.cat([
            self._ndvi(target),
            self._simple_ratio(target),
            self._green_swir(target),
        ], dim=1)

        sq_err = (indices_pred - indices_target) ** 2  # [B, 3, H, W]

        if mask is not None:
            cloud_area = torch.clamp(mask.sum(), min=1.0)
            loss = (sq_err * mask).sum() / (cloud_area * sq_err.shape[1])
        else:
            loss = sq_err.mean()

        return loss




class CombinedLoss(nn.Module):
    """
    Combined reconstruction loss for Sentinel-2 cloud removal:

        L = w_l1          x L1           (pixel MAE, cloud + 0.1 x clear)
          + w_ssim        x SSIM         (structural similarity, full image)
          + w_perceptual  x Perceptual   (VGG16 feature matching, cloud region)
          + w_charbonnier x Charbonnier  (smooth L1, edge-preserving, cloud)
          + w_spectral    x Spectral     (NDVI / SR / GSWIR consistency, cloud)

    Default weights: 0.5 / 0.3 / 0.1 / 0.1 / 0.05  (total = 1.05).
    The total > 1.0 is intentional — spectral is additive at a small scale.
    """

    def __init__(
        self,
        w_l1:          float = 0.5,
        w_ssim:        float = 0.3,
        w_perceptual:  float = 0.1,
        w_charbonnier: float = 0.1,
        w_spectral:    float = 0.05,
        n_bands:       int   = 6,
    ) -> None:
        super().__init__()
        self.w_l1          = w_l1
        self.w_ssim        = w_ssim
        self.w_perceptual  = w_perceptual
        self.w_charbonnier = w_charbonnier
        self.w_spectral    = w_spectral

        self.l1_loss          = nn.L1Loss(reduction="none")
        self.ssim_loss        = SSIMLoss(channel=n_bands)
        self.perceptual_loss  = PerceptualLoss()
        self.charbonnier_loss = CharbonnierLoss()
        self.spectral_loss    = SpectralConsistencyLoss()

    def forward(
        self,
        pred:   torch.Tensor,                   # [B, C, H, W]
        target: torch.Tensor,                   # [B, C, H, W]
        mask:   Optional[torch.Tensor] = None,  # [B, H, W] or [B, 1, H, W]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns (total_loss, component_dict) for logging.
        """
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)      # [B, 1, H, W]

            # 1. L1 — cloud-region primary + clear-region identity enforcement
            l1_map = self.l1_loss(pred, target)
            cloud_area = torch.clamp(mask.sum(), min=1.0)
            clear_mask = 1.0 - mask
            clear_area = torch.clamp(clear_mask.sum(), min=1.0)
            l1_cloud = (l1_map * mask).sum() / (cloud_area * l1_map.shape[1])
            l1_clear = (l1_map * clear_mask).sum() / (clear_area * l1_map.shape[1])
            l1 = l1_cloud + 0.1 * l1_clear

            # 2. Perceptual — cloud-masked VGG16 feature matching
            perc = self.perceptual_loss(pred, target, mask)

            # 3. Charbonnier — smooth L1, cloud-masked
            charb = self.charbonnier_loss(pred, target, mask)

            # 4. Spectral — NDVI / SR / Green-SWIR consistency, cloud-masked
            spec = self.spectral_loss(pred, target, mask)
        else:
            l1   = self.l1_loss(pred, target).mean()
            perc = self.perceptual_loss(pred, target, None)
            charb = self.charbonnier_loss(pred, target, None)
            spec = self.spectral_loss(pred, target, None)

        # 5. SSIM — sliding window over full image (no masking — spatial metric)
        ssim = self.ssim_loss(pred, target)

        total = (
            self.w_l1          * l1    +
            self.w_ssim        * ssim  +
            self.w_perceptual  * perc  +
            self.w_charbonnier * charb +
            self.w_spectral    * spec
        )

        components = {
            "loss/total":       total.item(),
            "loss/l1":          l1.item(),
            "loss/ssim":        ssim.item(),
            "loss/perceptual":  perc.item(),
            "loss/charbonnier": charb.item(),
            "loss/spectral":    spec.item(),
        }

        return total, components
