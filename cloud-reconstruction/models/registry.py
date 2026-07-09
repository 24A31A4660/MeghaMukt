"""models/registry.py — Model factory for cloud reconstruction architectures.

Maps config model names to their implementing classes.
Supports: "swin_unet" (default), "unet" (legacy).
"""
from __future__ import annotations

import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a model based on cfg["model"]["name"].

    Args:
        cfg: Full configuration dict (passed through to model constructor).

    Returns:
        Initialized nn.Module (not moved to device — caller handles that).

    Raises:
        ValueError: If model name is not recognized.
    """
    name = cfg["model"].get("name", "swin_unet").lower()

    if name == "swin_unet":
        from models.swin_unet import SwinUNet
        return SwinUNet(cfg)
    elif name == "unet":
        from models.unet import UNet
        return UNet(cfg)
    else:
        raise ValueError(
            f"Unknown model name '{name}'. Supported: 'swin_unet', 'unet'."
        )
