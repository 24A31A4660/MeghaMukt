"""models/losses.py — Combined reconstruction loss: L1 + SSIM + Perceptual + Edge."""
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available or self.features is None:
            return torch.tensor(0.0, device=pred.device)

        pred_rgb   = self._extract_rgb(pred)
        target_rgb = self._extract_rgb(target)

        pred_feat   = self.features(pred_rgb)
        target_feat = self.features(target_rgb)
        return F.l1_loss(pred_feat, target_feat, reduction='none')


# ─────────────────────────────────────────────────────────────────────────────
# Edge Loss (Sobel)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """
    Sobel gradient-aware loss.
    Penalises differences in gradient magnitude between prediction and target.
    Helps preserve roads, field boundaries, coastlines.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Shape: [1, 1, 3, 3] — applied channel-wise
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Sobel gradient magnitude for each channel.
        Input: [B, C, H, W]  Output: [B, C, H, W]
        """
        b, c, h, w = x.shape
        x_flat = x.view(b * c, 1, h, w)
        gx = F.conv2d(x_flat, self.kx, padding=1)
        gy = F.conv2d(x_flat, self.ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.view(b, c, h, w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad   = self._sobel(pred)
        target_grad = self._sobel(target)
        return F.l1_loss(pred_grad, target_grad, reduction='none')


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combined reconstruction loss:
        L = w_l1 × L1 + w_ssim × SSIM + w_perceptual × VGG + w_edge × Edge

    L1 loss is computed for both clear and cloud pixels (to enforce identity).
    Perceptual and Edge losses are computed strictly on cloud-masked pixels.
    All losses are evaluated on full tensors before spatial masking to avoid artificial boundaries.

    Default weights: 0.4 / 0.3 / 0.2 / 0.1
    """

    def __init__(
        self,
        w_l1:         float = 0.4,
        w_ssim:       float = 0.3,
        w_perceptual: float = 0.2,
        w_edge:       float = 0.1,
        n_bands:      int   = 6,
    ) -> None:
        super().__init__()
        self.w_l1         = w_l1
        self.w_ssim       = w_ssim
        self.w_perceptual = w_perceptual
        self.w_edge       = w_edge

        self.l1_loss          = nn.L1Loss(reduction="none")
        self.ssim_loss        = SSIMLoss(channel=n_bands)
        self.perceptual_loss  = PerceptualLoss()
        self.edge_loss        = EdgeLoss()

    def forward(
        self,
        pred:   torch.Tensor,    # [B, C, H, W]
        target: torch.Tensor,    # [B, C, H, W]
        mask:   Optional[torch.Tensor] = None,  # [B, H, W] or [B, 1, H, W]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns (total_loss, component_dict) for logging.
        """
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)      # [B, 1, H, W]

            # 1. L1 Loss
            l1_map = self.l1_loss(pred, target)
            cloud_area = torch.clamp(mask.sum(), min=1.0)
            clear_mask = 1.0 - mask
            clear_area = torch.clamp(clear_mask.sum(), min=1.0)
            
            # mask will broadcast across C channels for l1_map
            l1_cloud = (l1_map * mask).sum() / (cloud_area * l1_map.shape[1])
            l1_clear = (l1_map * clear_mask).sum() / (clear_area * l1_map.shape[1])
            l1 = l1_cloud + l1_clear

            # 2. Edge Loss
            edge_map = self.edge_loss(pred, target)
            edge_cloud = (edge_map * mask).sum() / (cloud_area * edge_map.shape[1])
            edge = edge_cloud

            # 3. Perceptual Loss
            perc_map = self.perceptual_loss(pred, target)
            if perc_map.dim() == 0:
                perc = perc_map
            else:
                mask_down = F.adaptive_avg_pool2d(mask, perc_map.shape[2:])
                cloud_area_down = torch.clamp(mask_down.sum(), min=1.0)
                perc_cloud = (perc_map * mask_down).sum() / (cloud_area_down * perc_map.shape[1])
                perc = perc_cloud
        else:
            l1 = self.l1_loss(pred, target).mean()
            edge = self.edge_loss(pred, target).mean()
            perc = self.perceptual_loss(pred, target).mean()

        # SSIM is computed globally (sliding window over full image)
        ssim = self.ssim_loss(pred, target)

        total = (
            self.w_l1         * l1   +
            self.w_ssim       * ssim +
            self.w_perceptual * perc +
            self.w_edge       * edge
        )

        components = {
            "loss/total":      total.item(),
            "loss/l1":         l1.item(),
            "loss/ssim":       ssim.item(),
            "loss/perceptual": perc.item(),
            "loss/edge":       edge.item(),
        }

        return total, components
