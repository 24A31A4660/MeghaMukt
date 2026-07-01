"""models/base_model.py — Abstract base class for all reconstruction models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base for all cloud reconstruction models.

    All models must implement:
        forward(cloudy, cloud_mask, sentinel1=None) → prediction
        get_config() → dict

    This ensures training/inference code works without modification
    when swapping between UNet, Pix2Pix, Diffusion, ViT, etc.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        cloudy: torch.Tensor,
        cloud_mask: torch.Tensor,
        sentinel1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cloudy      : [B, C_optical, H, W] — multi-band cloudy image.
        cloud_mask  : [B, 1, H, W] — binary cloud mask.
        sentinel1   : [B, C_sar, H, W] — optional SAR data (None if not available).

        Returns
        -------
        prediction  : [B, C_optical, H, W] — reconstructed image.
        """
        ...

    @abstractmethod
    def get_config(self) -> dict:
        """Return model configuration as a dict (for logging/checkpointing)."""
        ...

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (useful for fine-tuning)."""
        if hasattr(self, "encoder"):
            for p in self.encoder.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True
