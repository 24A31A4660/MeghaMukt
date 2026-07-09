"""utils/checkpoint.py — Save and load model checkpoints.

Supports two calling conventions:
  1. Structured: save_checkpoint(path, epoch, model, optimizer, scheduler, metrics, config, scaler)
  2. Dict-style: save_checkpoint(dict_payload, path)  ← used by train_optimized.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch


def save_checkpoint(
    path_or_payload: Union[Path, str, dict],
    epoch_or_path: Union[int, Path, str, None] = None,
    model: Optional["torch.nn.Module"] = None,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[dict[str, float]] = None,
    config: Optional[dict] = None,
    scaler_state_dict: Optional[dict] = None,
) -> None:
    """Save a full training checkpoint.

    Supports two calling styles:
        # Style 1 — structured (used by training/trainer.py)
        save_checkpoint(path, epoch, model, optimizer, scheduler, metrics, config, scaler)

        # Style 2 — dict payload (used by train_optimized.py)
        save_checkpoint(payload_dict, path)
    """
    if isinstance(path_or_payload, dict):
        # Dict-style call: save_checkpoint(payload, path)
        payload = path_or_payload
        path = Path(epoch_or_path) if epoch_or_path else Path("checkpoint.pth")
    else:
        # Structured call
        path = Path(path_or_payload)
        payload = {
            "epoch": epoch_or_path,
            "model_state_dict": model.state_dict() if model else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler_state_dict,
            "metrics": metrics or {},
            "config": config or {},
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint. Returns the payload dict.

    If strict=False, allows missing keys (useful for loading old checkpoints with new architectures).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    # Load model state dict with optional strict mode
    try:
        model.load_state_dict(payload["model_state_dict"], strict=strict)
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e) and not strict:
            model.load_state_dict(payload["model_state_dict"], strict=False)
            print(f"[Checkpoint] Loaded with missing keys (new layers initialized): {path}")
        else:
            raise

    if optimizer and payload.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except Exception as e:
            print(f"[Warning] Could not load optimizer state: {e}")

    if scheduler and payload.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        except Exception as e:
            print(f"[Warning] Could not load scheduler state: {e}")

    if scaler is not None and payload.get("scaler_state_dict"):
        try:
            scaler.load_state_dict(payload["scaler_state_dict"])
        except Exception as e:
            print(f"[Warning] Could not load scaler state: {e}")

    return payload


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recently modified checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Check for latest_checkpoint.pth first (new naming)
    for name in ("latest_checkpoint.pth", "latest.pth"):
        latest = checkpoint_dir / name
        if latest.exists():
            return latest

    # Fall back to epoch-numbered checkpoints
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return best checkpoint if it exists. Checks multiple naming conventions."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    for name in ("best_model.pth", "best.pth", "best_v2.pth"):
        best = checkpoint_dir / name
        if best.exists():
            return best
    return None
