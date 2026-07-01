"""evaluation/confidence.py — Per-pixel reconstruction confidence map."""
from __future__ import annotations

import numpy as np


def compute_confidence_map(
    prediction: np.ndarray,
    cloudy_input: np.ndarray,
    cloud_mask: np.ndarray,
    sigma: float = 0.1,
) -> np.ndarray:
    """
    Compute a per-pixel confidence map for the reconstruction.

    Confidence reflects how much the prediction differs from the cloudy input
    in cloud-masked regions. High confidence = prediction is far from the noisy
    cloud pixel, suggesting the model is reconstructing rather than copying.

    Formula (cloud pixels):
        confidence = exp(-|prediction - cloudy| / sigma)

    Clear pixels always have confidence = 1.0 (unchanged by model).

    Parameters
    ----------
    prediction   : [C, H, W] float32 in [0, 1] — model output.
    cloudy_input : [C, H, W] float32 in [0, 1] — input image.
    cloud_mask   : [H, W] binary uint8 — 1=cloud, 0=clear.
    sigma        : Scaling factor for the exponential decay.

    Returns
    -------
    confidence : [H, W] float32 in [0, 1].
    """
    # Mean absolute difference across bands
    diff = np.mean(np.abs(prediction - cloudy_input), axis=0)  # [H, W]

    # Exponential confidence: 1 = no change (clear), decays as change increases
    confidence = np.exp(-diff / sigma)

    # Cloud-free pixels → confidence = 1.0 (model doesn't touch them)
    confidence[cloud_mask == 0] = 1.0

    return confidence.astype(np.float32)


def compute_difference_map(
    prediction: np.ndarray,
    target: np.ndarray,
    amplify: float = 3.0,
) -> np.ndarray:
    """
    Compute a visualisable difference map between prediction and ground truth.

    Parameters
    ----------
    prediction : [C, H, W] float32.
    target     : [C, H, W] float32.
    amplify    : Scale factor to make differences more visible.

    Returns
    -------
    diff_map : [H, W] float32 in [0, 1], amplified for visualisation.
    """
    diff = np.mean(np.abs(prediction - target), axis=0)  # [H, W]
    diff = np.clip(diff * amplify, 0.0, 1.0)
    return diff.astype(np.float32)
