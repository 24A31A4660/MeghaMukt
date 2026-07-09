"""models/unet.py — U-Net with Sentinel-1 FusionGate extension point."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F




# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive Conv→BN→ReLU layers."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool downsampling."""

    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, use_batchnorm)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)       # [B, out_ch, H, W]
        down = self.pool(skip)    # [B, out_ch, H/2, W/2]
        return down, skip


class AttentionGate(nn.Module):
    """Lightweight gate that re-weights skip features before decoder fusion."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, max(1, in_ch // 2), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_ch // 2), 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = self.proj(x)
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return skip * gate


class DecoderBlock(nn.Module):
    """Transposed conv upsampling + attention-gated skip connection + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(in_ch // 2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, use_batchnorm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-1 Fusion Gate
# ─────────────────────────────────────────────────────────────────────────────

class FusionGate(nn.Module):
    """
    Cross-modal fusion gate between optical bottleneck features and Sentinel-1 SAR.

    When Sentinel-1 is disabled: acts as identity (zero overhead).
    When enabled: learned attention-weighted feature fusion.

    Extension point: set sentinel1_enabled=True and provide SAR channels.
    """

    def __init__(
        self,
        optical_ch: int,
        sar_ch: int = 0,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        if enabled and sar_ch > 0:
            self.sar_proj = nn.Conv2d(sar_ch, optical_ch, kernel_size=1)
            self.gate = nn.Sequential(
                nn.Conv2d(optical_ch * 2, optical_ch, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.sar_proj = None
            self.gate = None

    def forward(
        self,
        optical_feat: torch.Tensor,
        sar_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled or sar_feat is None or self.gate is None:
            return optical_feat  # identity passthrough

        sar_proj = self.sar_proj(sar_feat)
        sar_proj = F.interpolate(sar_proj, size=optical_feat.shape[2:], mode="bilinear", align_corners=False)
        gate_weight = self.gate(torch.cat([optical_feat, sar_proj], dim=1))
        return optical_feat + gate_weight * sar_proj


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for cloud reconstruction.

    Input:
        cloudy    : [B, C_optical, H, W]  — 6 optical bands
        cloud_mask: [B, 1, H, W]          — binary cloud mask
        (Concatenated internally → [B, C_optical+1, H, W])

    Output:
        prediction: [B, C_optical, H, W]  — reconstructed bands

    Architecture:
        Encoder × depth levels → Bottleneck → Decoder × depth levels
        Skip connections between matching encoder/decoder levels
        FusionGate between encoder and bottleneck for Sentinel-1 (future)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg


        in_ch       = cfg["model"]["input_channels"]   # 7 (6 bands + mask)
        out_ch      = cfg["model"]["output_channels"]  # 6
        base_f      = cfg["model"]["base_filters"]     # 64
        depth       = cfg["model"]["depth"]            # 4
        use_bn      = cfg["model"]["use_batchnorm"]
        dropout     = cfg["model"]["dropout"]
        s1_enabled  = cfg.get("sentinel1", {}).get("enabled", False)
        s1_channels = cfg.get("sentinel1", {}).get("channels", 2)

        # Compute channel sizes per level
        filters = [base_f * (2 ** i) for i in range(depth)]  # [64, 128, 256, 512]

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_ch
        for f in filters:
            self.encoders.append(EncoderBlock(ch, f, use_bn))
            ch = f

        # Bottleneck
        bn_ch = filters[-1] * 2
        self.bottleneck = ConvBlock(ch, bn_ch, use_bn, dropout)

        # Fusion gate (Sentinel-1 extension)
        self.fusion_gate = FusionGate(bn_ch, s1_channels, enabled=s1_enabled)

        # Decoder
        self.decoders = nn.ModuleList()
        ch = bn_ch
        for f in reversed(filters):
            self.decoders.append(DecoderBlock(ch, f, f, use_bn))
            ch = f

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(ch, out_ch, kernel_size=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(
        self,
        cloudy: torch.Tensor,
        cloud_mask: torch.Tensor,
        sentinel1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Concatenate cloud mask if required by input_channels config
        expected_in_ch = self.cfg["model"]["input_channels"]
        if expected_in_ch > cloudy.shape[1]:
            if len(cloud_mask.shape) == 3:
                cloud_mask = cloud_mask.unsqueeze(1)
            x = torch.cat([cloudy, cloud_mask], dim=1)  # [B, C+1, H, W]
        else:
            x = cloudy  # [B, C, H, W]

        # Encoder path
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Sentinel-1 fusion (identity if disabled)
        x = self.fusion_gate(x, sentinel1)

        # Decoder path
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)

    def get_config(self) -> dict:
        return {
            "model": "UNet",
            "input_channels":  self.cfg["model"]["input_channels"],
            "output_channels": self.cfg["model"]["output_channels"],
            "base_filters":    self.cfg["model"]["base_filters"],
            "depth":           self.cfg["model"]["depth"],
            "parameters":      self.count_parameters(),
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
