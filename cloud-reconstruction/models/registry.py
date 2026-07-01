"""models/registry.py — Model registry for plug-and-play model swapping."""
from __future__ import annotations

from typing import Type

import torch.nn as nn

from models.base_model import BaseModel
from models.unet import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Stub classes for future models
# ─────────────────────────────────────────────────────────────────────────────

class Pix2Pix(BaseModel):
    """STUB — Pix2Pix GAN model. Replace with full implementation."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "Pix2Pix is not yet implemented. Set model.name=unet in config.yaml."
        )

    def forward(self, cloudy, cloud_mask, sentinel1=None):
        raise NotImplementedError

    def get_config(self) -> dict:
        return {"model": "Pix2Pix", "status": "stub"}


class DiffusionModel(BaseModel):
    """STUB — Diffusion model. Replace with full implementation."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "DiffusionModel is not yet implemented. Set model.name=unet in config.yaml."
        )

    def forward(self, cloudy, cloud_mask, sentinel1=None):
        raise NotImplementedError

    def get_config(self) -> dict:
        return {"model": "Diffusion", "status": "stub"}


class VisionTransformer(BaseModel):
    """STUB — Vision Transformer (Swin-T based). Replace with full implementation."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        raise NotImplementedError(
            "ViT is not yet implemented. Set model.name=unet in config.yaml."
        )

    def forward(self, cloudy, cloud_mask, sentinel1=None):
        raise NotImplementedError

    def get_config(self) -> dict:
        return {"model": "ViT", "status": "stub"}


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "unet":      UNet,
    "pix2pix":   Pix2Pix,
    "diffusion": DiffusionModel,
    "vit":       VisionTransformer,
}


def build_model(cfg: dict) -> BaseModel:
    """
    Instantiate the model specified in config.yaml under model.name.

    Usage:
        model = build_model(cfg)

    Switching models: change config.yaml model.name and call this function.
    No other code changes required.
    """
    name = cfg["model"]["name"].lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](cfg)


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(MODEL_REGISTRY.keys())
