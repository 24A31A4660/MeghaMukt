"""utils/checkpoint.py — Save and load model checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    path: Path,
    epoch: int,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: Optional[Any],
    metrics: dict[str, float],
    config: dict,
) -> None:
    """Save a full training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint. Returns the payload dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler and payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return payload


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recently modified checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return best.pth if it exists."""
    best = Path(checkpoint_dir) / "best.pth"
    return best if best.exists() else None
